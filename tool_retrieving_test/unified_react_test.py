import json
import statistics
import sys
import tempfile
from pathlib import Path
from httpx import Client

import dotenv
import regex as re
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from config import (
    BENCHMARK_PATH,
    TOOLS_PATH,
    SUBTASK_K,
    TOP_M,
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    HTTP_PROXY,
    PLANNER_AGENT_SYSTEM_PROMPT,
    AGENT_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
)
from tool_utils import (
    load_benchmark,
    load_tools,
    format_tool_descriptions,
)

from schemas import ToolSchema

dotenv.load_dotenv()

PROGRESS_FILE = Path("results/tmp_unified_progress_top10.json")


class STEmbeddings(Embeddings):
    """Adapter so LangChain can use SentenceTransformers inside Chroma."""

    def __init__(
        self, st_model: SentenceTransformer, *, query_prompt_name: str = "query"
    ):
        self.model = st_model
        self.query_prompt_name = query_prompt_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [text], prompt_name=self.query_prompt_name, normalize_embeddings=False
        )[0]
        return vec.tolist()


def build_docs(
    tool_names: set[str], tools_schema: dict[str, ToolSchema]
) -> list[Document]:
    """Convert each tool into a LangChain Document for vector search."""
    docs: list[Document] = []
    for name in tool_names:
        schema = tools_schema[name]
        args_schema = schema.arguments
        if args_schema and args_schema.properties:
            arg_strings = [
                f"{arg}: {prop.type} — {prop.description or ''}"
                for arg, prop in args_schema.properties.items()
            ]
            flat_args = " | ".join(arg_strings)
        else:
            flat_args = "none"
        page_text = (
            f"{name}\n"
            f"{schema.description_expanded}\n"
            f"Arguments: {flat_args}\n"
            f"Synthetic questions: {schema.synthetic_questions}"
        )
        docs.append(Document(page_content=page_text, metadata={"tool_name": name}))
    return docs


def compute_metrics(ref: set[str], selected: set[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for one query."""
    tp = len(ref & selected)
    precision = tp / len(selected) if selected else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def parse_called_tools(text: str) -> set[str]:
    """
    Extract tool names from the agent JSON in `text`.
    Expected chunk is a list of dicts with `"tool": "<name>"`.
    """
    try:
        m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if not m:
            return set()
        data = json.loads(m.group(0))
        if not isinstance(data, list):
            return set()
        return {
            item.get("tool")
            for item in data
            if isinstance(item, dict) and item.get("tool")
        }
    except Exception:
        return set()


def invoke_planner(
    llm: ChatOpenAI, user_request: str, tool_desc_block: str
) -> set[str]:
    """Return a set of subtasks produced by the planner LLM."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                PLANNER_AGENT_SYSTEM_PROMPT
                + "\n\nAvailable tools:\n"
                + tool_desc_block,
            )
        ]
    ).format_prompt(user_request=user_request)
    response = llm.invoke(prompt).content
    start, end = response.find("["), response.rfind("]")
    json_str = response[start : end + 1] if start != -1 and end != -1 else "[]"
    try:
        subtasks = json.loads(json_str)
        return set(subtasks) if isinstance(subtasks, list) else set()
    except Exception:
        return set()


def invoke_argument_agent(
    llm: ChatOpenAI,
    user_request: str,
    selected_tool_names: set[str],
    tools_schema: dict[str, ToolSchema],
) -> list[str]:
    """Identify missing arguments via a single LLM call.

    Returns a list of new sub-questions (one per missing argument).
    """
    if not selected_tool_names:
        return []

    lines: list[str] = []
    for name in sorted(selected_tool_names):
        schema = tools_schema.get(name)
        if not schema:
            continue
        args_schema = schema.arguments
        if not (args_schema and args_schema.properties):
            continue
        arg_strings = [
            f"{arg}: {prop.type} — {prop.description or ''}"
            for arg, prop in args_schema.properties.items()
        ]
        flat_args = " | ".join(arg_strings)

        # arg_list = ", ".join(args_schema.properties.keys())
        lines.append(f"{name}: {flat_args}")

    if not lines:
        return []

    tools_and_args_block = "\n".join(lines)

    prompt = ChatPromptTemplate.from_messages(
        [("human", REACT_AGENT_SYSTEM_PROMPT)]
    ).format_prompt(
        user_request=user_request,
        tools_and_args=tools_and_args_block,
    )
    resp = llm.invoke(prompt).content

    start, end = resp.find("["), resp.rfind("]")
    json_str = resp[start : end + 1] if start != -1 and end != -1 else "[]"
    try:
        items = json.loads(json_str)
        return (
            [s for s in items if isinstance(s, str)] if isinstance(items, list) else []
        )
    except Exception:
        return []


def invoke_agent(
    llm: ChatOpenAI,
    user_request: str,
    subtask: str,
    subset_schema: dict[str, ToolSchema],
) -> str:
    """Single call of the tool-calling agent."""
    desc_block = format_tool_descriptions(subset_schema)
    prompt = ChatPromptTemplate.from_messages(
        [("human", AGENT_SYSTEM_PROMPT)]
    ).format_prompt(
        tool_descriptions=desc_block,
        current_subtask=subtask,
        user_request=user_request,
    )
    return llm.invoke(prompt).content


def main() -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    tools_schema = load_tools(TOOLS_PATH)
    if not benchmark or not tools_schema:
        sys.exit("Failed to load benchmark or tools JSON.")

    st_model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
    )
    st_model.max_seq_length = 8192
    embeddings = STEmbeddings(st_model)

    all_docs = build_docs(set(tools_schema), tools_schema)
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="full_tools",
        persist_directory=tempfile.mkdtemp(prefix="full_tools_vdb_"),
    )

    llm = ChatOpenAI(
        base_url=OPENROUTER_URL,
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_KEY,
        temperature=0,
        http_client=Client(proxy=HTTP_PROXY),
    )

    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        completed: set[int] = set(data.get("completed", []))
        detail_table = data.get("detail_table", [])
    else:
        completed = set()
        detail_table = []

    for idx, item in enumerate(benchmark):
        if idx in completed:
            continue

        user_request = item.question
        ref_tools = {r.tool for r in item.reference if r.tool in tools_schema}

        retrieved_docs = vectordb.similarity_search(user_request, k=TOP_M)
        planner_tools = {
            doc.metadata["tool_name"]: tools_schema[doc.metadata["tool_name"]]
            for doc in retrieved_docs
            if doc.metadata["tool_name"] in tools_schema
        }
        tool_desc_block = format_tool_descriptions(planner_tools)

        subtasks = invoke_planner(llm, user_request, tool_desc_block)

        selected_tools: set[str] = set()

        for subtask in subtasks:
            retrieved_docs = vectordb.similarity_search(subtask, k=SUBTASK_K)
            subtask_tools = {doc.metadata["tool_name"] for doc in retrieved_docs}

            if not subtask_tools:
                subtask_tools = set(planner_tools.keys())

            sub_schema = {name: tools_schema[name] for name in subtask_tools}

            resp = invoke_agent(
                llm=llm,
                user_request=user_request,
                subtask=subtask,
                subset_schema=sub_schema,
            )

            selected_tools.update(parse_called_tools(resp))

        react_subtasks = invoke_argument_agent(
            llm=llm,
            user_request=user_request,
            selected_tool_names=selected_tools,
            tools_schema=tools_schema,
        )

        for subtask in react_subtasks:
            retrieved_docs = vectordb.similarity_search(subtask, k=SUBTASK_K)
            subtask_tools = {doc.metadata["tool_name"] for doc in retrieved_docs}

            if not subtask_tools:
                subtask_tools = set(planner_tools.keys())

            sub_schema = {
                name: tools_schema[name]
                for name in subtask_tools
                if name in tools_schema
            }

            resp = invoke_agent(
                llm=llm,
                user_request=user_request,
                subtask=subtask,
                subset_schema=sub_schema,
            )

            selected_tools.update(parse_called_tools(resp))

        precision, recall, f1 = compute_metrics(ref_tools, selected_tools)

        detail_table.append(
            {
                "idx": idx,
                "question": user_request,
                "reference": sorted(ref_tools),
                "selected": sorted(selected_tools),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        completed.add(idx)

        PROGRESS_FILE.write_text(
            json.dumps(
                {"completed": sorted(completed), "detail_table": detail_table},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{idx:04d}] P={precision:.3f} R={recall:.3f} F1={f1:.3f}  "
            f"refs={len(ref_tools)} sel={len(selected_tools)}"
        )

    if not detail_table:
        print("No items processed.")
        return

    precisions = [r["precision"] for r in detail_table]
    recalls = [r["recall"] for r in detail_table]
    f1s = [r["f1"] for r in detail_table]

    print("\n===== Unified RAG + LLM benchmark =====")
    print(f"Total questions: {len(detail_table)}")
    print(
        f"Precision={statistics.mean(precisions):.3f} | "
        f"Recall={statistics.mean(recalls):.3f} | "
        f"F1={statistics.mean(f1s):.3f}"
    )


if __name__ == "__main__":
    main()
