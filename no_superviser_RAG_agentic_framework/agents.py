import json
import tempfile
import uuid
from typing import Dict, List, Optional, Any

import regex as re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from pydantic import ValidationError

from no_superviser_RAG_agentic_framework.config import (
    AGENT_SYSTEM_PROMPT,
    LLM_MODEL,
    OPENAI_API_KEY,
    TOP_RANK
)
from no_superviser_RAG_agentic_framework.schemas import (
    AgentState,
    ToolCall,
    ToolSchema
)
from no_superviser_RAG_agentic_framework.tool_utils import (
    format_tool_descriptions,
    validate_tool_call_sequence,
)


def create_agent():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{user_request}"),
        ]
    )

    def _build_vectorstore(tools_dict: Dict[str, ToolSchema]) -> Chroma:
        tmp_dir = tempfile.mkdtemp(prefix=f"chroma_rag_tools_")
        docs: List[Document] = []

        for tool_name, schema in tools_dict.items():
            arg_strings = []
            args_schema = schema.arguments
            if args_schema and args_schema.properties:
                for arg_name, arg_prop in args_schema.properties.items():
                    arg_strings.append(
                        f"{arg_name}: {arg_prop.type} — {arg_prop.description or ''}"
                    )
            flat_args = " | ".join(arg_strings) if arg_strings else "none"

            page_text = f"{tool_name}\n{schema.description}\nArguments: {flat_args}"
            docs.append(
                Document(page_content=page_text, metadata={"tool_name": tool_name})
            )

        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=f"{uuid.uuid4().hex}",
            persist_directory=tmp_dir,
        )

    def agent_node(state: AgentState) -> Dict[str, Any]:
        print(f"\n--- Agent Node ---")
        print(f"User Request: {state.user_request}")
        if state.error_message:
            print(f"Previous Error: {state.error_message}")

        current_agent_tools = state.get_agent_tools()

        vectordb = _build_vectorstore(current_agent_tools)

        # retrieval_query = f"{state.user_request} {state.error_message or ''}".strip()
        retrieval_query = state.user_request.strip()
        # print("!!!!!", retrieval_query)
        try:
            retrieved_docs = vectordb.similarity_search(retrieval_query, k=TOP_RANK)
            ranked_names = [
                doc.metadata.get("tool_name")
                for doc in retrieved_docs
                if doc.metadata.get("tool_name") in current_agent_tools
            ]
        except Exception as e:
            print(f"Vector search failed: {e}")
            ranked_names = []

        if not ranked_names:
            ranked_names = list(current_agent_tools.keys())[:TOP_RANK]

        unique_tool_names: set = set()
        top_tool_names = [
            name
            for name in ranked_names
            if not (name in unique_tool_names or unique_tool_names.add(name))
        ]
        # print("!!!!!", top_tool_names)
        top_tools: Dict[str, ToolSchema] = {
            name: current_agent_tools[name] for name in top_tool_names
        }

        tool_desc_string = format_tool_descriptions(top_tools)

        # Format the prompt with current state
        formatted_prompt = prompt_template.format_prompt(
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
            update_dict["agent_outcome"] = []
            update_dict["error_message"] = (
                state.error_message
                or error_parsing
                or "Agent failed to produce valid tool calls."
            )
        token_usage = response.response_metadata.get('token_usage', {})
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        update_dict['total_prompt_tokens'] = prompt_tokens
        update_dict['total_completion_tokens'] = completion_tokens

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
