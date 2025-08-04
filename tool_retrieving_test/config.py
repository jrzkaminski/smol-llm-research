import os

TOOLS_PATH = "../data/ultratool/tools_expanded.json"
BENCHMARK_PATH = "../data/ultratool/top_benchmarks_enriched.json"
SEED = 42

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"

K = 10
SUBTASK_K = 5


PLANNER_AGENT_SYSTEM_PROMPT = """
Rewrite the USER REQUEST as the smallest sequence of independent, solvable sub-requests.

Context: you have access to 15 tools provided in a separate context section (via RAG). Use them to
think how the request can be decomposed so that every sub-request can be solved by SOME tool
from the list. Split as aggressively as possible – the more fine-grained the better (but preserve
logical order). 

Rules:
1. Each sub-request must be a self-contained natural-language instruction; no code or tool names.
2. Preserve order and any quoted literals (file names, texts, numbers).
3. Return ONLY a JSON array of strings. No keys, no commentary.

User request: "{user_request}"
"""


AGENT_SYSTEM_PROMPT = """
You are an execution agent working on a SINGLE sub-task: \"{current_subtask}\".
Full-task: \"{user_request}\"

Available tools:
{tool_descriptions}

Select the best tool(s) (one or many or all) to accomplish the sub-task. For tools that perform the same
high-level action (e.g. file_write vs create_document) include ALL candidates.

Return ONLY a JSON array of tool calls following the format:
[
  {{
    "tool": "tool_name",
    "param": {{ ... }},
    "input_source": "question" | "<prev_tool> tool"
  }},
  ...
]

Do NOT output any commentary.
"""
