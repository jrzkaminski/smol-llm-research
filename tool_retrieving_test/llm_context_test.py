import json
import tempfile
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import dotenv
import regex as re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import (
    AGENT_SYSTEM_PROMPT,
    LLM_MODEL,
    OPENAI_API_KEY,
    PLANNER_AGENT_SYSTEM_PROMPT,
    BENCHMARK_PATH,
    TOOLS_PATH,
)
from tool_utils import (
    load_benchmark,
    load_tools,
    format_tool_descriptions,
)
from schemas import ToolSchema

dotenv.load_dotenv()

PROGRESS_FILE = Path("tmp_llm_context_progress.json")


def compute_metrics(ref: Set[str], selected: Set[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for one query."""
    tp = len(ref & selected)
    precision = tp / len(selected) if selected else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def parse_called_tools(text: str) -> Set[str]:
    """
    Extract a set of tool names from the agent’s JSON output in `text`.
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
    query: str, subtasks: Set[str], subset_schema: Dict[str, ToolSchema], llm: ChatOpenAI
) -> str:
    desc_block = format_tool_descriptions(subset_schema)
    prompt = ChatPromptTemplate.from_messages(
        [("human", AGENT_SYSTEM_PROMPT)]
    ).format_prompt(tool_descriptions=desc_block, user_request=query, subtasks=subtasks, error="None")
    return llm.invoke(prompt).content


def build_docs(
    tool_names: Set[str], tools_schema: Dict[str, ToolSchema]
) -> List[Document]:
    """Build a list of LangChain Document objects for the given tool names."""
    docs: List[Document] = []
    for name in tool_names:
        schema = tools_schema[name]
        page_text = get_tool_doc(name, schema)
        docs.append(Document(page_content=page_text, metadata={"tool_name": name}))
    return docs


def get_tool_doc(tool_name: str, schema: ToolSchema) -> str:
    args_schema = schema.arguments
    if args_schema and args_schema.properties:
        arg_strings = [
            f"{arg}: {prop.type} — {prop.description or ''}"
            for arg, prop in args_schema.properties.items()
        ]
        flat_args = " | ".join(arg_strings)
    else:
        flat_args = "none"
    return f"{tool_name}\n{schema.description}\nArguments: {flat_args}"


def get_noise_tools(ref_tools: Set[str], count: int, tools_schema: Dict[str, ToolSchema], collection: Chroma) -> List[str]:
    if count == 0:
        return []
    buffer = set()
    result = []
    for name in ref_tools:
        tool_doc = get_tool_doc(name, tools_schema[name])
        searches = collection.similarity_search_with_score(query=tool_doc, k=count)
        for doc, similarity in searches:
            buffer.add((doc.metadata['tool_name'], similarity))
    buffer = sorted(list(buffer), key=lambda x: x[1])
    for tool_name, similarity in buffer:
        if tool_name not in ref_tools:
            result.append(tool_name)
    return result[:count]


def invoke_planner(
    query: str, llm: ChatOpenAI
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("human", PLANNER_AGENT_SYSTEM_PROMPT)]
    ).format_prompt(user_request=query, error="None")
    return llm.invoke(prompt).content


def parse_planner_subtasks(json_string: str) -> Set[str]:
    result = json.loads(json_string)
    return result


def run_single_eval(
    query: str,
    ref_tools: Set[str],
    tools_schema: Dict[str, ToolSchema],
    noise_level: int,
    llm: ChatOpenAI,
    subtasks: Set[str],
    collection: Chroma
) -> Tuple[Set[str], float, float, float]:
    """Return (selected_set, precision, recall, f1)."""
    noise_pool = list(set(tools_schema) - ref_tools)
    count = min(noise_level, len(noise_pool))
    noise_tools = get_noise_tools(ref_tools=ref_tools, count=count, tools_schema=tools_schema, collection=collection)
    names: List[str] = list(ref_tools.union(noise_tools))
    subset_schema = {n: tools_schema[n] for n in names}

    response = invoke_agent(query, subtasks, subset_schema, llm)
    selected = parse_called_tools(response)

    precision, recall, f1 = compute_metrics(ref_tools, selected)
    return selected, precision, recall, f1


def main() -> None:

    bench = load_benchmark(BENCHMARK_PATH)
    tools_schema = load_tools(TOOLS_PATH)
    if not bench or not tools_schema:
        sys.exit("Failed to load benchmark or tools JSON.")

    noise_levels = list(range(0, 51, 10))

    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        completed: Set[int] = set(data.get("completed", []))
        detail_table: Dict[int, List[Dict]] = {
            int(k): v for k, v in data.get("detail_table", {}).items()
        }
        for n in noise_levels:
            detail_table.setdefault(n, [])
    else:
        completed = set()
        detail_table = {n: [] for n in noise_levels}

    documents = build_docs(set(tools_schema), tools_schema)
    all_tools_vectordb = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        collection_name="rag_ref_tools",
        persist_directory=tempfile.mkdtemp(prefix="rag_ref_tools_"),
    )

    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    for idx, item in enumerate(bench):
        if idx in completed:
            continue

        ref = {r.tool for r in item.reference if r.tool in tools_schema}
        if not ref:
            completed.add(idx)
            continue

        query = item.question
        response = invoke_planner(query=query, llm=llm)
        subtasks = parse_planner_subtasks(response)

        for noise in noise_levels:
            selected, prec, rec, f1 = run_single_eval(
                query=query,
                ref_tools=ref,
                tools_schema=tools_schema,
                noise_level=noise,
                llm=llm,
                collection=all_tools_vectordb,
                subtasks=subtasks
            )
            detail_table[noise].append(
                {
                    "idx": idx,
                    "ref": sorted(ref),
                    "selected": sorted(selected),
                    "precision": prec,
                    "recall": rec,
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

    print("\n===== LLM context tool-calling benchmark =====")
    print(f"total questions: {len(completed)}")
    print("noise level -> presicion | recall | f1")

    for noise in noise_levels:
        recs = detail_table[noise]
        if not recs:
            print(f"{noise:2d}: 0.000 | 0.000 | 0.000")
            continue

        precisions = [
            (
                r.get("precision")
                if "precision" in r
                else compute_metrics(set(r["ref"]), set(r["selected"]))[0]
            )
            for r in recs
        ]
        recalls = [
            (
                r.get("recall")
                if "recall" in r
                else compute_metrics(set(r["ref"]), set(r["selected"]))[1]
            )
            for r in recs
        ]
        f1s = [
            (
                r.get("f1")
                if "f1" in r
                else compute_metrics(set(r["ref"]), set(r["selected"]))[2]
            )
            for r in recs
        ]

        mp = statistics.mean(precisions)
        mr = statistics.mean(recalls)
        mf = statistics.mean(f1s)
        print(f"{noise:2d}: {mp:.3f} | {mr:.3f} | {mf:.3f}")


if __name__ == "__main__":
    main()
