import os

GRAPH_JSON_PATH = "../data/ultratool/graph.json"
TOOLS_JSON_PATH = "../data/ultratool/tools.json"
BENCHMARK_JSON_PATH = "../data/ultratool/benchmarks.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o-mini"

TOP_RANK = 10


PLANNER_AGENT_SYSTEM_PROMPT = """
You are a task-planner agent for an autonomous tool-execution system.

Below are the tools most relevant to the current request.
Other tools exist, but build your plan using ONLY the tools listed here.

{tool_block}

Rules for the plan you will output:

1. Generate the SHORTEST ordered list of subtasks that, if done in sequence, fully solve the USER REQUEST below.  
2. Write every subtask as a **bare infinitive clause** (verb first) that already contains ALL required argument names and values.  
Do NOT include tool names, JSON keys, or any stop words such as “the”, “a”, “an”, “please”, “should”, “need to”, “kindly”.  
3. If one tool can handle the request, output exactly ONE subtask.  
4. Preserve order and every literal from the user (dates, codes, names, numbers, quoted text) in each subtask.  
5. Return ONLY a valid JSON array of strings, e.g.  
   [
     "Retrieve stock trend for code 600519 from 2025-06-16 to 2025-06-17",
     "Predict future trend for code 600519 using historical data above"
   ]
"""

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

AGENT_SYSTEM_PROMPT = """You are a specialist agent responsible for the '{category}' category of tools.
You have access to the following tools:
{tool_descriptions}

Relevant relationships between these tools (directed as 'source -> target'):
{links_description}

Your task is to process the user request: "{user_request}"
You must select the appropriate tool(s) from *your* available list and determine the correct arguments to fulfill the request.
You need to output a list of proposed tool calls. Each tool call should be a dictionary with 'tool' (the tool name) and 'param' (a dictionary of arguments).

For the 'input_source' field:
- Use "question" if the required information is present in the user request/question
- Use "tool_name tool" if the required information comes from the output of a previous tool that should be run first (e.g., "file_write tool", "account_login tool")

If you CANNOT complete the entire task, IN ANY CASE, WRITE DOWN THE FUNCTIONS that, in your opinion, CAN BRING you CLOSER to solving the problem in JSON format.
Use tools ONLY FROM THE LIST PROVIDED TO YOU!
If the function is not available to you, DO NOT WRITE it in JSON.
Make sure that your response in JSON format is correct!!
Return ONLY a JSON. No keys, no commentary. At least one tool must be in it.

Example Output Format:
[
  {{
    "tool": "tool_name_1",
    "param": {{
      "arg_name_1": "value_1",
      "arg_name_2": "value_2"
    }},
    "input_source": "question"
  }},
  {{
    "tool": "tool_name_2",
    "param": {{
      "arg_name_3": "value_3"
    }},
    "input_source": "tool_name_1 tool"
  }}
]

If you receive an error message about an invalid tool call, analyze the error and try again with corrected tool name or arguments.
Error: {error}

Based on the request and your available tools, propose the sequence of tool calls.
"""
