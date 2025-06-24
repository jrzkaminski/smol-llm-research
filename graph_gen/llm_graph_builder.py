import json
import itertools
from tqdm import tqdm
import torch
import re
import os
import warnings

import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
# Suppress a specific warning from the transformers library for cleaner output.
warnings.filterwarnings("ignore", message="Passing `pad_token_id` to `eos_token_id`")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Dependencies not found. Please install them:")
    print("pip install torch transformers tqdm accelerate")
    exit()

PROGRESS_PATH = "progress_edges.json"
SAVE_EVERY = 100


def load_progress(path=PROGRESS_PATH):
    """
    Returns:
        processed_edges (set[frozenset]): edges already evaluated
        edges_json      (list[dict])    : [{'source': ..., 'target': ...}, ...]
    """
    if not os.path.isfile(path):
        return set(), []
    with open(path, "r") as f:
        data = json.load(f)
    edges_json = data.get("edges", [])
    processed_edges = {frozenset([e["source"], e["target"]]) for e in edges_json}
    return processed_edges, edges_json


def save_progress(edges_json, path=PROGRESS_PATH):
    """Serialize the current edge list to disk."""
    with open(path, "w") as f:
        json.dump({"edges": edges_json}, f, indent=2)


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets of undirected edges."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def load_llm(model_id="google/gemma-3-4b-it"):
    """Loads the LLM and tokenizer, setting the device to MPS."""
    print("Loading LLM... This may take a few minutes.")

    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU. This will be very slow.")
        device = "cpu"
    else:
        device = "cuda"
        print("CUDA is available. Using GPU for acceleration.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    return model, tokenizer, device


def sanitize_tool(tool):
    """Removes the 'depends_on' key from a tool's dictionary."""
    if isinstance(tool, dict):
        # Create a copy to avoid modifying the original dictionary in memory
        tool_copy = tool.copy()
        if "depends_on" in tool_copy:
            del tool_copy["depends_on"]
        return tool_copy
    return tool


def query_llm(prompt, model, tokenizer, device):
    """Queries the LLM and parses the True/False response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a response. We don't need a long response.
    outputs = model.generate(
        **inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response part
    response_only = response_text[len(prompt) :].strip()

    # Look for 'True' or 'False' case-insensitively
    if re.search(r"true", response_only, re.IGNORECASE):
        return True
    if re.search(r"false", response_only, re.IGNORECASE):
        return False

    # Fallback if the model doesn't respond as expected
    # print(f"\n[Warning] LLM did not return a clear True/False. Response: '{response_only}'. Defaulting to False.")
    return False


def build_llm_graph():
    """Builds and evaluates the tool graph using an LLM."""
    # --- 1. Load Data ---
    print("Loading toollinkos data...")
    with open("data/toollinkos/core_tools.json", "r") as f:
        core_tools = json.load(f)
    with open("data/toollinkos/regular_tools.json", "r") as f:
        regular_tools = json.load(f)

    all_tools = core_tools + regular_tools
    print(f"Loaded {len(all_tools)} tools in total.")

    # --- 2. Extract Ground Truth ---
    ground_truth_edges = set()
    for tool in all_tools:
        tool_name = tool["name"]
        for dependency in tool.get("depends_on", []):
            dep_name = dependency["name"]
            # Create an undirected edge using frozenset
            edge = frozenset([tool_name, dep_name])
            ground_truth_edges.add(edge)
    print(f"Found {len(ground_truth_edges)} ground truth edges.")

    processed_edges, edges_json = load_progress()
    print(f"Resuming: {len(processed_edges)} pairs already done.")

    # --- 3. Load LLM ---
    model, tokenizer, device = load_llm()

    # --- 4. Define Prompt Template ---
    prompt_template = """
You are a highly intelligent assistant that analyzes software tools to determine if they depend on each other.

Here are the types of dependencies:
I. Tool Directly Depends On: A tool requires another tool to operate. Example: "set_wifi_on" must be executed before a tool requiring internet.
II. Tool Indirectly Depends On: A tool benefits from another but does not strictly require it. Example: a "restaurant_reservation" tool may use "get_weather", but can function without it.
III. Parameter Directly Depends On: A required parameter must be obtained from another tool. Example: "product_info" needs a "product_id" from "get_product_id".
IV. Parameter Indirectly Depends On: A parameter depends on additional context only if required by the user. Example: "tomorrow" in a scheduling tool requires "get_current_date".

Based on these definitions, analyze the two tools below. Does Tool A depend on Tool B, or does Tool B depend on Tool A?

Tool A:
{tool_a}

Tool B:
{tool_b}

Is there any type of dependency between Tool A and Tool B? Answer with only the word "True" or "False".
"""

    # --- 5. Main Loop ---
    predicted_edges = set(processed_edges)
    tool_pairs = list(itertools.combinations(all_tools, 2))
    print(f"\nStarting analysis of {len(tool_pairs)} tool pairs...")
    for tool_a, tool_b in tqdm(tool_pairs, desc="Analyzing tool pairs"):
        # Sanitize tools to remove ground truth
        sanitized_a = sanitize_tool(tool_a)
        sanitized_b = sanitize_tool(tool_b)

        # Format the tools as clean JSON strings for the prompt
        tool_a_str = json.dumps(sanitized_a, indent=2)
        tool_b_str = json.dumps(sanitized_b, indent=2)

        prompt = prompt_template.format(tool_a=tool_a_str, tool_b=tool_b_str)

        if query_llm(prompt, model, tokenizer, device):
            edge = frozenset([tool_a["name"], tool_b["name"]])
            predicted_edges.add(edge)

            edges_json.append({"source": list(edge)[0], "target": list(edge)[1]})

            if len(predicted_edges) % SAVE_EVERY == 0:
                save_progress(edges_json)
                curr_jaccard = jaccard_similarity(ground_truth_edges, predicted_edges)
                print(
                    f"[Checkpoint] {len(predicted_edges)} edges — "
                    f"Jaccard={curr_jaccard:.4f}",
                    flush=True,
                )
    # --- 6. Evaluation ---
    jaccard = jaccard_similarity(ground_truth_edges, predicted_edges)
    save_progress(edges_json)
    print("\n--- LLM-based Evaluation Results ---")
    print(f"Ground Truth Edges: {len(ground_truth_edges)}")
    print(f"Predicted Edges (from LLM): {len(predicted_edges)}")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print("------------------------------------")


if __name__ == "__main__":
    build_llm_graph()
