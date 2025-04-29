import json
import time

from langgraph.graph import StateGraph, END

from RAG_agentic_framework.agents import create_agent, validation_node
from RAG_agentic_framework.config import (
    GRAPH_JSON_PATH,
    TOOLS_JSON_PATH,
    BENCHMARK_JSON_PATH,
)
from RAG_agentic_framework.schemas import AgentState  # Ensure ToolCall is imported
from RAG_agentic_framework.supervisor import (
    create_supervisor_router,
    supervisor_aggregator_node,
)
from RAG_agentic_framework.tool_utils import (
    load_graph,
    load_tools,
    load_benchmark,
    get_tools_by_category,
)

print("--- Loading Setup Data ---")
graph_structure = load_graph(GRAPH_JSON_PATH)
all_tools_schema = load_tools(TOOLS_JSON_PATH)
benchmark_items = load_benchmark(BENCHMARK_JSON_PATH)

benchmark_items = benchmark_items[0:100]


if not graph_structure or not all_tools_schema or not benchmark_items:
    print("Error loading necessary graph, tools, or benchmark data. Exiting.")
    exit()

tools_by_category = get_tools_by_category(graph_structure, all_tools_schema)
categories = list(tools_by_category.keys())

# --- Build Graph ---
print("--- Building Agent Graph ---")
workflow = StateGraph(AgentState)

agent_nodes = {}
for category, tools in tools_by_category.items():
    agent_nodes[category] = create_agent()
    workflow.add_node(category, agent_nodes[category])

# Add Validation Node
workflow.add_node("validation", validation_node)

# Add Supervisor Nodes
supervisor_router_node = create_supervisor_router(categories)  # Get the node function
workflow.add_node("supervisor_router", supervisor_router_node)
workflow.add_node("supervisor_aggregator", supervisor_aggregator_node)
workflow.add_node(
    "error_end",
    lambda state: print(
        f"Ending graph for request due to error: {state.get('error_message', 'Unknown error')}"
    )
    or {},
)


# --- Define Edges ---
workflow.set_entry_point("supervisor_router")

workflow.add_conditional_edges(
    "supervisor_router",
    lambda state: state.category if state.category in categories else "error_end",
    {cat: cat for cat in categories} | {"error_end": "error_end"},
)

for category in categories:
    workflow.add_edge(category, "validation")


def decide_after_validation(state: AgentState) -> str:
    if state.error_message:
        print(f"Validation failed, routing back to agent {state.category} for retry.")
        return state.category
    else:
        print("Validation succeeded, proceeding to supervisor aggregator.")
        return "supervisor_aggregator"


workflow.add_conditional_edges(
    "validation",
    decide_after_validation,
    {cat: cat for cat in categories}
    | {"supervisor_aggregator": "supervisor_aggregator"},
)

workflow.add_edge("supervisor_aggregator", END)
workflow.add_edge("error_end", END)


# --- Compile Graph ---
print("--- Compiling Graph ---")
app = workflow.compile()

# --- Run Benchmarks ---
print(f"\n--- Running {len(benchmark_items)} Benchmarks ---")
all_results = []
app_config = {"recursion_limit": 10}

for i, item in enumerate(benchmark_items):
    print(f"\n--- Starting Benchmark {i+1}/{len(benchmark_items)} ---")
    print(f"Question: {item.question}")

    initial_state = AgentState(
        user_request=item.question,
        all_tools_schema=all_tools_schema,
        tools_by_category=tools_by_category,
        category=None,
        agent_outcome=None,
        error_message=None,
        supervisor_result=None,
        total_prompt_tokens=0,
        total_completion_tokens=0,
    )

    final_state = {}
    try:
        final_state = app.invoke(initial_state, config=app_config)

        print("\n--- Graph Execution Finished for Benchmark ---")
        generated_calls_list = final_state.get("supervisor_result", [])
        error_msg = final_state.get("error_message")
        final_prompt_tokens = final_state.get("total_prompt_tokens", 0)
        final_completion_tokens = final_state.get("total_completion_tokens", 0)
        final_cost = final_state.get("total_cost", 0.0)
        print(f"Final Supervisor Result (Raw Objects): {generated_calls_list}")
        print(
            f"Final Tokens - Prompt: {final_prompt_tokens}, Completion: {final_completion_tokens}"
        )
        if error_msg:
            print(f"Run finished with error: {error_msg}")

    except Exception as e:
        print(f"\n--- Error during graph execution for benchmark {i+1} ---")
        print(f"Error: {e}")
        generated_calls_list = []
        error_msg = f"Graph execution failed: {e}"
        final_prompt_tokens = 0
        final_completion_tokens = 0
        final_cost = 0.0

    # --- Store results for this benchmark ---
    generated_calls_dicts = [call.model_dump() for call in generated_calls_list]
    reference_calls_dicts = [ref.model_dump() for ref in item.reference]

    result_data = {
        "id": item.id or f"benchmark_{i+1}",
        "question": item.question,
        "generated_calls": generated_calls_dicts,
        "reference_calls": reference_calls_dicts,
        "final_error": error_msg,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
    }
    all_results.append(result_data)

    time.sleep(1)

# --- Save All Results ---
output_file = "results_all.json"
print(f"\n--- Saving all {len(all_results)} results to {output_file} ---")
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved successfully.")
except Exception as e:
    print(f"\nError saving results to JSON: {e}")

print("\n--- Main Script Finished ---")
