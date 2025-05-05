from typing import List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_framework.config import SUPERVISOR_SYSTEM_PROMPT, LLM_MODEL, OPENAI_API_KEY
from agentic_framework.schemas import AgentState


def create_supervisor_router(categories: List[str]):
    """
    Creates a conditional routing function for the supervisor.
    This node determines which agent (category) to delegate to and updates the state.
    """
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
        ]
    )

    def supervisor_router(
        state: AgentState,
    ) -> Dict[str, Any]:
        print("\n--- Supervisor Router Node ---")
        print(f"User Request: {state.user_request}")
        print(f"Available Categories: {categories}")

        formatted_prompt = prompt.format_prompt(
            categories=", ".join(categories), user_request=state.user_request
        )
        print(f"Routing prompt sent to LLM:\n{formatted_prompt.to_string()}")

        response: BaseMessage = llm.invoke(formatted_prompt)
        chosen_category_text = (
            response.content.strip()
        )

        prompt_tokens = 0
        completion_tokens = 0
        try:
            token_usage = response.response_metadata.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)
            print(f"Supervisor Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        except Exception as e:
            print(f"Warning: Could not extract token usage from supervisor response: {e}")
        print(f"LLM suggested category text: {chosen_category_text}")

        best_match = None
        for cat in categories:
            if cat.lower() == chosen_category_text.lower():
                best_match = cat
                break
        if not best_match:
            for cat in categories:
                if cat.lower() in chosen_category_text.lower():
                    best_match = cat
                    break

        update_dict = {}
        if best_match:
            print(f"Routing to agent: {best_match}")
            update_dict["category"] = best_match
        else:
            print(
                "Error: Supervisor could not determine a valid category from LLM response."
            )
            update_dict["category"] = None
            update_dict["error_message"] = (
                f"Supervisor could not route based on LLM suggestion: '{chosen_category_text}'"
            )
        update_dict["total_prompt_tokens"] = state.total_prompt_tokens + prompt_tokens
        update_dict["total_completion_tokens"] = state.total_completion_tokens + completion_tokens

        return update_dict

    return supervisor_router


def supervisor_aggregator_node(state: AgentState) -> Dict[str, Any]:
    """Node where the supervisor receives the final validated outcome from the agent."""
    print("\n--- Supervisor Aggregator Node ---")
    update_dict = {}
    if state.agent_outcome:
        print(
            f"Received validated tool calls from agent ({state.category}): {state.agent_outcome}"
        )
        update_dict["supervisor_result"] = state.agent_outcome
        update_dict["error_message"] = None
    else:
        print("No valid outcome received from agent flow.")
        update_dict["supervisor_result"] = []
        if not state.error_message:
            update_dict["error_message"] = "Agent could not complete the task."
        else:
            update_dict["error_message"] = state.error_message
    return update_dict
