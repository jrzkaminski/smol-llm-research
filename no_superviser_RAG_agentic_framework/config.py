import os

GRAPH_JSON_PATH = "../data/ultratool/graph.json"
TOOLS_JSON_PATH = "../data/ultratool/tools.json"
BENCHMARK_JSON_PATH = "../data/ultratool/benchmarks.json"

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

LLM_MODEL = "gpt-4o"

TOP_RANK = 10

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent. Your role is to understand the user's request \
and delegate it to the appropriate specialist agent based on the required tool category.
The available agent categories are: {categories}.
Analyze the user's request: "{user_request}".
Determine which category of tools is most relevant to fulfill the request.
Delegate the task to the agent responsible for that category.
You will receive the proposed tool calls from the agent.
Your final output should be the sequence of tool calls proposed by the agent.
If the request seems to require tools from multiple categories, decide the primary category or sequence them if applicable (though current setup focuses on single delegation).
Output only the final list of tool calls in the specified format.
"""

AGENT_SYSTEM_PROMPT = """You are a specialist agent responsible for tools.
You have access to the following tools:
{tool_descriptions}

Your task is to process the user request: "{user_request}"
You must select the appropriate tool(s) from *your* available list and determine the correct arguments to fulfill the request.
You need to output a list of proposed tool calls. Each tool call should be a dictionary with 'tool' (the tool name) and 'param' (a dictionary of arguments).

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
