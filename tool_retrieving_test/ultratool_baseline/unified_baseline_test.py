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

from simple_config import (
    BENCHMARK_PATH,
    TOOLS_PATH,
    TASK_K,
    HTTP_PROXY,
    AGENT_SYSTEM_PROMPT,
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
)
from tool_utils import (
    load_benchmark,
    load_tools,
    simple_format_tool_descriptions,
)

from schemas import ToolSchema

dotenv.load_dotenv()

PROGRESS_FILE = Path("results/tmp_unified_baseline_progress.json")


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
        page_text = f"{name}\n" f"{schema.description}\n" f"Arguments: {flat_args}\n"
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


def invoke_agent(
    llm: ChatOpenAI,
    user_request: str,
    subset_schema: dict[str, ToolSchema],
) -> str:
    """Single call of the tool-calling agent."""
    desc_block = simple_format_tool_descriptions(subset_schema)
    prompt = ChatPromptTemplate.from_messages(
        [("human", AGENT_SYSTEM_PROMPT)]
    ).format_prompt(
        tool_descriptions=desc_block,
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
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_URL,
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

        selected_tools: set[str] = set()

        retrieved_docs = vectordb.similarity_search(user_request, k=TASK_K)
        request_tools = {doc.metadata["tool_name"] for doc in retrieved_docs}

        sub_schema = {name: tools_schema[name] for name in request_tools}

        resp = invoke_agent(
            llm=llm,
            user_request=user_request,
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
