import json
from typing import Dict, List, Optional, Any

import regex as re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from agentic_framework.config import AGENT_SYSTEM_PROMPT, LLM_MODEL, OPENAI_API_KEY
from agentic_framework.schemas import AgentState, ToolCall
from agentic_framework.tool_utils import (
    format_tool_descriptions,
    validate_tool_call_sequence,
)


def create_agent():
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{user_request}"),
        ]
    )

    def agent_node(state: AgentState) -> Dict[str, Any]:
        print(f"\n--- Agent Node ({state.category}) ---")
        print(f"User Request: {state.user_request}")
        if state.error_message:
            print(f"Previous Error: {state.error_message}")

        current_agent_tools = state.get_current_agent_tools()

        tool_desc_string = format_tool_descriptions(
            current_agent_tools
        )

        # Format the prompt with current state
        formatted_prompt = prompt_template.format_prompt(
            category=state.category,
            tool_descriptions=tool_desc_string,
            user_request=state.user_request,
            error=state.error_message or "None",
        )

        print(f"Prompt sent to LLM:\n{formatted_prompt.to_string()}")

        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        response_content = response.content
        print(f"LLM Response:\n{response_content}")

        parsed_outcome: Optional[List[ToolCall]] = None
        error_parsing = None
        try:
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                validated_calls = [
                    ToolCall.model_validate(item) for item in parsed_data
                ]
                parsed_outcome = validated_calls
                print(f"Successfully parsed tool calls: {parsed_outcome}")
            else:
                error_parsing = (
                    "Could not find valid JSON list of tool calls in the LLM response."
                )
                print(error_parsing)

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            error_parsing = f"Failed to parse or validate LLM response into ToolCall list: {e}. Response was: {response_content}"
            print(error_parsing)

        update_dict = {}
        if parsed_outcome:
            update_dict["agent_outcome"] = parsed_outcome
            update_dict["error_message"] = None
        else:
            update_dict["agent_outcome"] = None
            update_dict["error_message"] = (
                state.error_message
                or error_parsing
                or "Agent failed to produce valid tool calls."
            )

        return update_dict

    return agent_node



def validation_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- Validation Node ---")
    update_dict = {}

    agent_outcome = state.agent_outcome

    if agent_outcome is None:
        print("No agent outcome to validate.")
        if not state.error_message:
            update_dict["error_message"] = (
                "Agent did not produce a valid list of tool calls."
            )
        return update_dict

    print(f"Validating: {agent_outcome}")

    # Perform validation
    is_valid, error = validate_tool_call_sequence(agent_outcome, state.all_tools_schema)

    if not is_valid:
        print(f"Validation Failed: {error}")
        update_dict["error_message"] = error
        update_dict["agent_outcome"] = None
    else:
        print("Validation Successful.")
        update_dict["error_message"] = None

    return update_dict
