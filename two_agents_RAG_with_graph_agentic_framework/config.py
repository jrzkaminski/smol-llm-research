import os

GRAPH_JSON_PATH = "../data/ultratool/graph.json"
TOOLS_JSON_PATH = "../data/ultratool/tools.json"
BENCHMARK_JSON_PATH = "../data/ultratool/benchmarks.json"

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

LLM_MODEL = "gpt-4o-mini"

TOP_RANK = 10
SUBTASK_TOP_RANK = 5

PLANNER_AGENT_SYSTEM_PROMPT = """
Your task is to process the user request: "{user_request}"
You need to extract subtasks from the user's question. Return the list of subtasks in the following format:
["subtask_1", "subtask_2", "subtask_3"]
"""

AGENT_SYSTEM_PROMPT = """You are a specialized agent for performing user tasks. You have a set of tools for that.
You have access to the following tools:
{tool_descriptions}

Your task is to process the user request: "{user_request}"
You must select the appropriate tool(s) from *your* available list and determine the correct arguments to fulfill the request.
You need to output a list of proposed tool calls. Each tool call should be a dictionary with 'tool' (the tool name) and 'param' (a dictionary of arguments).
If you CANNOT complete the entire task, IN ANY CASE, WRITE DOWN THE FUNCTIONS that, in your opinion, CAN BRING you CLOSER to solving the problem in JSON format.
Use tools ONLY FROM THE LIST PROVIDED TO YOU!
If the function is not available to you, DO NOT WRITE it in JSON.
Make sure that your response in JSON format is correct!!

Example Output Format:
[
  {{
    "tool": "tool_name_1",
    "param": {{
      "arg_name_1": "value_1",
      "arg_name_2": "value_2"
    }}
  }},
  {{
    "tool": "tool_name_2",
    "param": {{
      "arg_name_3": "value_3"
    }}
  }}
]

If you receive an error message about an invalid tool call, analyze the error and try again with corrected tool name or arguments.
Error: {error}

Based on the request and your available tools, propose the sequence of tool calls.
"""
