import os
from dotenv import load_dotenv

load_dotenv()

TOOLS_PATH = "../../data/ultratool/tools_expanded.json"
BENCHMARK_PATH = "../../data/ultratool/top_benchmarks_enriched.json"
SEED = 42

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY")
LLM_MODEL = "gpt-4o-mini"

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

TASK_K = 5


AGENT_SYSTEM_PROMPT = """
You are an execution agent working on the task: \"{user_request}\"

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
