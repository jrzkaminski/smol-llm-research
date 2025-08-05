import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


TOOLS_PATH = Path("data/ultratool/tools.json")
OUTPUT_PATH = Path("data/ultratool/tools_expanded.json")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set. Please export your key before running."
    )

MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("OPENAI_RETRY_DELAY", 1))

client = OpenAI(api_key=API_KEY)


def load_existing() -> list[dict[str, Any]]:
    """Load existing expanded tools list (if file exists)."""
    if not OUTPUT_PATH.exists():
        return []
    try:
        with OUTPUT_PATH.open() as f:
            return json.load(f)
    except Exception:
        return []


def already_processed(existing: list[dict[str, Any]]) -> set[str]:
    """Return a set with names of tools already present in existing list."""
    return {t.get("name") for t in existing if isinstance(t, dict)}


def call_llm(prompt: str) -> str:
    """Call the LLM with basic retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"[Retry {attempt}/{MAX_RETRIES}] LLM error: {e}")
            time.sleep(RETRY_DELAY)


def build_prompt(tool: dict[str, Any], n_questions: int = 5) -> str:
    """Generate a prompt asking the model to enrich description and propose synthetic questions."""
    arguments = json.dumps(tool.get("arguments", {}), ensure_ascii=False)
    results = json.dumps(tool.get("results", {}), ensure_ascii=False)
    return (
        "Expand the following tool description in 5 sentences, focusing on what the tool does, typical use-cases, and important details.\n"
        f"Then propose {n_questions} diverse user requests that could be solved *using only this tool*.\n\n"
        "Return the response strictly as valid JSON without additional text or code fences. Use this format:\n"
        '{ "name": str, "description_expanded": str, "synthetic_questions": [str, ...] }\n\n'
        f"Tool name: {tool['name']}\n"
        f"Original description: {tool['description']}\n"
        f"Arguments schema: {arguments}\n"
        f"Results schema: {results}\n"
    )


def main(n_questions_per_tool: int = 5) -> None:
    with TOOLS_PATH.open() as f:
        tools: list[dict[str, Any]] = json.load(f)

    existing_expanded: list[dict[str, Any]] = load_existing()
    processed = already_processed(existing_expanded)
    total = len(tools)
    print(f"Total tools: {total}. Already processed: {len(processed)}.")

    for idx, tool in enumerate(tools, 1):
        name = tool.get("name")
        if name in processed:
            continue

        print(f"[{idx}/{total}] Processing: {name}")
        prompt = build_prompt(tool, n_questions_per_tool)

        raw = call_llm(prompt)
        try:
            obj = json.loads(raw)
            if not (
                obj.get("name")
                and obj.get("description_expanded")
                and obj.get("synthetic_questions")
            ):
                raise ValueError("Missing required fields in LLM response")

            merged_tool: dict[str, Any] = {**tool, **obj}
        except Exception as e:
            print(
                f"Failed to parse or validate response for tool '{name}': {e}. Skipping."
            )
            continue

        existing_expanded.append(merged_tool)
        with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
            json.dump(existing_expanded, out_f, ensure_ascii=False, indent=4)

        time.sleep(0.2)


if __name__ == "__main__":
    main(n_questions_per_tool=5)
