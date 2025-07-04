import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import dotenv
import regex as re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    AGENT_SYSTEM_PROMPT,
    LLM_MODEL,
    OPENAI_API_KEY,
    SEED,
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
    query: str, subset_schema: Dict[str, ToolSchema], llm: ChatOpenAI
) -> str:
    desc_block = format_tool_descriptions(subset_schema)
    prompt = ChatPromptTemplate.from_messages(
        [("human", AGENT_SYSTEM_PROMPT)]
    ).format_prompt(tool_descriptions=desc_block, user_request=query, error="None")
    return llm.invoke(prompt).content


def run_single_eval(
    query: str,
    ref_tools: Set[str],
    tools_schema: Dict[str, ToolSchema],
    noise_level: int,
    rng: random.Random,
    llm: ChatOpenAI,
) -> Tuple[Set[str], float, float, float]:
    """Return (selected_set, precision, recall, f1)."""
    noise_pool = list(set(tools_schema) - ref_tools)
    noise_tools = rng.sample(noise_pool, k=min(noise_level, len(noise_pool)))
    names: List[str] = list(ref_tools.union(noise_tools))
    rng.shuffle(names)
    subset_schema = {n: tools_schema[n] for n in names}

    response = invoke_agent(query, subset_schema, llm)
    selected = parse_called_tools(response)

    precision, recall, f1 = compute_metrics(ref_tools, selected)
    return selected, precision, recall, f1


def main() -> None:

    bench = load_benchmark(BENCHMARK_PATH)
    tools_schema = load_tools(TOOLS_PATH)
    if not bench or not tools_schema:
        sys.exit("Failed to load benchmark or tools JSON.")

    rng = random.Random(SEED)
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

    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    for idx, item in enumerate(bench):
        if idx in completed:
            continue

        ref = {r.tool for r in item.reference if r.tool in tools_schema}
        if not ref:
            completed.add(idx)
            continue

        for noise in noise_levels:
            selected, prec, rec, f1 = run_single_eval(
                query=item.question,
                ref_tools=ref,
                tools_schema=tools_schema,
                noise_level=noise,
                rng=rng,
                llm=llm,
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
