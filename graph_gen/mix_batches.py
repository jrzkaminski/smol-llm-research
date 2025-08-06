import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", message="Passing `pad_token_id` to `eos_token_id`")


load_dotenv()
DEEPSEEK_KEY: str = os.getenv("DEEPSEEK_KEY", "")

DATA_DIR = Path("data/toollinkos")
CHECKPOINT_PATH = Path("graph_gen/llm_mixed_checkpoint.json")
RESULTS_PATH = Path("graph_gen/llm_mixed_results.json")

BATCH_SIZE = 25
SAVE_EVERY = 1


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def sanitize_tool(tool: dict) -> dict:
    """Return a copy without the transient key `depends_on`."""
    t = tool.copy()
    t.pop("depends_on", None)
    return t


def split_into_primary_batches(tools: List[dict]) -> List[List[dict]]:
    """Shuffle *tools* and split into chunks of ≤ 100 tools."""
    random.shuffle(tools)
    return [tools[i : i + BATCH_SIZE] for i in range(0, len(tools), BATCH_SIZE)]


def create_mixed_batches(primary: List[List[dict]]) -> List[List[dict]]:
    """Generate mixed batches following (i ≥ j) schedule, each of size 100."""
    mixed: List[List[dict]] = []
    n = len(primary)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                batch = primary[i][:]
                random.shuffle(batch)
                mixed.append(batch)
            else:
                union = {t["name"]: t for t in (primary[i] + primary[j])}.values()
                selected = random.sample(list(union), min(BATCH_SIZE, len(union)))
                random.shuffle(selected)
                mixed.append(list(selected))
    return mixed


def parse_llm_response(resp: str, names: set) -> Dict[frozenset, str]:
    """Extract edges with rationale {frozenset(src,tgt): why}."""
    edges: Dict[frozenset, str] = {}
    try:
        match = re.search(r"\[.*?\]", resp, re.DOTALL)
        if not match:
            return edges
        items = json.loads(match.group(0))
        for item in items:
            src, tgt, why = item.get("source"), item.get("target"), item.get("why", "")
            if src in names and tgt in names and src != tgt:
                edges[frozenset((src, tgt))] = why
    except Exception:
        pass
    return edges


def build_mixed_llm_graph() -> None:
    print("\n── Mixed‑Batch LLM Graph Builder ──")

    # 1. Resume if checkpoint exists
    edge_reasons: Dict[frozenset, str] = {}
    last_done = -1
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        last_done = data.get("last_processed_index", -1)
        for obj in data.get("predicted_edges", []):
            edge_reasons[frozenset(obj[:2])] = obj[2]
        print(f"Resuming from batch {last_done + 1}; edges so far: {len(edge_reasons)}")

    # 2. Load tools & ground truth
    core_tools = json.loads((DATA_DIR / "core_tools.json").read_text())
    regular_tools = json.loads((DATA_DIR / "regular_tools.json").read_text())
    all_tools = core_tools + regular_tools
    print(f"Loaded {len(all_tools)} tools.")

    ground_truth = {
        frozenset((t["name"], d["name"]))
        for t in all_tools
        for d in t.get("depends_on", [])
    }
    print(f"Ground‑truth edges: {len(ground_truth)}")

    # 3. Primary and mixed batches
    primary_batches = split_into_primary_batches(all_tools)
    print(f"Primary batches: {len(primary_batches)} (≤100 each)")
    mixed_batches = create_mixed_batches(primary_batches)
    print(f"Mixed batches created: {len(mixed_batches)}")

    prompt_tpl = """Here are the types of dependencies to look for:
I. Tool Directly Depends On: A tool requires another to operate. (e.g., "set_wifi_on" before a tool needing internet).
II. Tool Indirectly Depends On: A tool benefits from another but doesn't strictly require it (e.g., "restaurant_reservation" might use "get_weather").
III. Parameter Directly Depends On: A required parameter must be obtained from another tool (e.g., "product_info" needs a "product_id").
IV. Parameter Indirectly Depends On: A parameter depends on context only if needed (e.g., "tomorrow" requires "get_current_date").

Analyze the list of tools provided below. Identify ALL pairs of tools (A, B) from this list where Tool A depends on Tool B, or Tool B depends on Tool A.

Return your ALL the findings as a single JSON array of objects, where each object represents a dependency with explanation.

Format: `[{{"source": "the dependent tool", "target": "the tool depends on", "why": "a concise one‑sentence rationale indicating"}}, {{"source": "the dependent tool1", "target": "the tool1 depends on", "why": "a concise one‑sentence rationale indicating"}}]`

Tools:
{tool_batch}
"""
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    for idx, batch in enumerate(tqdm(mixed_batches, desc="LLM batches")):
        if idx <= last_done:
            continue
        names = {t["name"] for t in batch}
        prompt = prompt_tpl.format(
            tool_batch=json.dumps([sanitize_tool(t) for t in batch], indent=2)
        )

        resp = (
            client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly intelligent assistant that analyzes a list of software tools to map their dependencies.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            .choices[0]
            .message.content
        )

        edge_reasons.update(parse_llm_response(resp, names))

        # Checkpoint
        if idx % SAVE_EVERY == 0 or idx == len(mixed_batches) - 1:
            ckpt_payload = {
                "last_processed_index": idx,
                "predicted_edges": [list(e) + [why] for e, why in edge_reasons.items()],
            }
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.write_text(json.dumps(ckpt_payload))
            print(
                f"[Checkpoint] batch={idx} edges={len(edge_reasons)} Jaccard={jaccard_similarity(ground_truth, set(edge_reasons)):.4f}",
                flush=True,
            )

    # 6. Final evaluation
    final_j = jaccard_similarity(ground_truth, set(edge_reasons))
    print("\n── Evaluation ──")
    print(f"Edge count (truth) : {len(ground_truth)}")
    print(f"Edge count (LLM)   : {len(edge_reasons)}")
    print(f"Jaccard similarity : {final_j:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "ground_truth_edges_count": len(ground_truth),
                "predicted_edges_count": len(edge_reasons),
                "jaccard_similarity": final_j,
                "predicted_edges": [list(e) + [why] for e, why in edge_reasons.items()],
            },
            indent=2,
        )
    )
    print(f"Results saved → {RESULTS_PATH}")

    # 7. Cleanup
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint removed.")


if __name__ == "__main__":
    build_mixed_llm_graph()
