import json
import time

from langgraph.graph import StateGraph, END

from agents import (
    create_agent,
    validation_node,
    create_planner
)
from config import (
    GRAPH_JSON_PATH,
    TOOLS_JSON_PATH,
    BENCHMARK_JSON_PATH,
)
from schemas import AgentState

from tool_utils import (
    load_graph,
    load_tools,
    load_benchmark,
    get_tools_by_category,
)

import dotenv

dotenv.load_dotenv()

print("--- Loading Setup Data ---")
graph_structure = load_graph(GRAPH_JSON_PATH)
all_tools_schema = load_tools(TOOLS_JSON_PATH)
benchmark_items = load_benchmark(BENCHMARK_JSON_PATH)

benchmark_items = benchmark_items

if not graph_structure or not all_tools_schema or not benchmark_items:
    print("Error loading necessary graph, tools, or benchmark data. Exiting.")
    exit()

tools_by_category = get_tools_by_category(graph_structure, all_tools_schema)
categories = list(tools_by_category.keys())

# --- Build Graph ---
print("--- Building Agent Graph ---")
workflow = StateGraph(AgentState)

agent = create_agent()
planner = create_planner()
workflow.add_node('planner', planner)
workflow.add_node('agent', agent)

workflow.add_edge('planner', "agent")

# Add Validation Node
workflow.add_node("validation", validation_node)
workflow.add_edge('agent', "validation")

workflow.add_node(
    "error_end",
    lambda state: print(
        f"Ending graph for request due to error: {state.get('error_message', 'Unknown error')}"
    )
    or {},
)

# --- Define Edges ---
workflow.set_entry_point("planner")


def decide_after_validation(state: AgentState) -> str:
    if state.error_message:
        print(f"Validation failed, routing back to agent {state.category} for retry.")
        return state.category
    else:
        print("Validation succeeded, proceeding to supervisor aggregator.")
        return "supervisor_aggregator"


workflow.add_edge("agent", END)
workflow.add_edge("error_end", END)

# --- Compile Graph ---
print("--- Compiling Graph ---")
app = workflow.compile()

# --- Run Benchmarks ---
print(f"\n--- Running {len(benchmark_items)} Benchmarks ---")
all_results = []
app_config = {"recursion_limit": 10}

for i, item in enumerate(benchmark_items):
    print(f"\n--- Starting Benchmark {i + 1}/{len(benchmark_items)} ---")
    print(f"Question: {item.question}")

    initial_state = AgentState(
        user_request=item.question,
        all_tools_schema=all_tools_schema,
        tools_by_category=tools_by_category,
        agent_outcome=None,
        error_message=None,
        subtasks=[],  # Explicitly initialize empty list
        total_prompt_tokens=0,
        total_completion_tokens=0,
        tools_graph=graph_structure,
        retry_count=0,  # Initialize retry_count for this benchmark
    )

    final_state = {}
    try:
        final_state = app.invoke(initial_state, config=app_config)

        print("\n--- Graph Execution Finished for Benchmark ---")
        generated_calls_list = final_state.get("agent_outcome", [])
        error_msg = final_state.get("error_message")
        final_prompt_tokens = final_state.get('total_prompt_tokens', 0)
        final_completion_tokens = final_state.get('total_completion_tokens', 0)
        final_cost = final_state.get('total_cost', 0.0)
        final_retry_count = final_state.get("retry_count", 0)
        print(f"Final Supervisor Result (Raw Objects): {generated_calls_list}")
        print(f"Final Tokens - Prompt: {final_prompt_tokens}, Completion: {final_completion_tokens}")
        print(f"Final Retries: {final_retry_count}")
        if error_msg:
            print(f"Run finished with error: {error_msg}")

    except Exception as e:
        print(f"\n--- Error during graph execution for benchmark {i + 1} ---")
        print(f"Error: {e}")
        generated_calls_list = []
        error_msg = f"Graph execution failed: {e}"
        final_prompt_tokens = 0
        final_completion_tokens = 0
        final_cost = 0.0
        # Inherit retry_count from initial_state if execution fails early
        final_retry_count = initial_state.retry_count

    # --- Store results for this benchmark ---
    if generated_calls_list is None:
        generated_calls_list = []
    generated_calls_dicts = [call.model_dump() for call in generated_calls_list]
    reference_calls_dicts = [ref.model_dump() for ref in item.reference]

    result_data = {
        "id": item.id or f"benchmark_{i + 1}",
        "question": item.question,
        "generated_calls": generated_calls_dicts,
        "reference_calls": reference_calls_dicts,
        "final_error": error_msg,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "retries_spent": final_retry_count
    }
    all_results.append(result_data)

    time.sleep(1)

# --- Save All Results ---
output_file = "results_all_4o_mini.json"
print(f"\n--- Saving all {len(all_results)} results to {output_file} ---")
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved successfully.")
except Exception as e:
    print(f"\nError saving results to JSON: {e}")

print("\n--- Main Script Finished ---")
