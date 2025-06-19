import json
from typing import List
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from RAG_graph_agentic_framework.tool_pruner import get_top_tool_schemas
from RAG_graph_agentic_framework.tool_utils import format_tool_descriptions
from RAG_graph_agentic_framework.config import PLANNER_AGENT_SYSTEM_PROMPT


def run_planner(
    user_request: str,
    llm: ChatOpenAI,
    top_k: int = 20,
) -> List[str]:
    candidate_tools = get_top_tool_schemas(user_request, top_k)
    tool_block = format_tool_descriptions(candidate_tools)

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_AGENT_SYSTEM_PROMPT),
            ("human", "{user_request}"),
        ]
    )

    response = llm.invoke(
        planner_prompt.format(tool_block=tool_block, user_request=user_request)
    )
    clean_response = re.sub(r"^```json\s*|```$", "", response.content.strip())
    subtasks = json.loads(clean_response)
    print(f"[PLANNER] request: {user_request}")
    print(f"[PLANNER] subtasks: {subtasks}")
    return subtasks if isinstance(subtasks, list) else [user_request]
