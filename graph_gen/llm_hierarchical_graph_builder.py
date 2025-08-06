import json
import random
from tqdm import tqdm
import torch
import re
import warnings
import os

# Suppress specific warnings for cleaner output.
warnings.filterwarnings("ignore", message="Passing `pad_token_id` to `eos_token_id`")
import torch._dynamo
from openai import OpenAI

torch._dynamo.config.cache_size_limit = 64
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Dependencies not found. Please install them:")
    print("pip install torch transformers tqdm accelerate")
    exit()

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")

# --- Constants ---
CHECKPOINT_PATH = "graph_gen/llm_hierarchical_checkpoint.json"
RESULTS_PATH = "graph_gen/llm_hierarchical_results.json"
SAVE_EVERY = 1


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets of undirected edges."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


# def load_llm(model_id="google/gemma-3-4b-it"):
#     """Loads the LLM and tokenizer, setting the device to MPS."""
#     print(f"Loading LLM: {model_id}... This may take a few minutes.")

#     if not torch.cuda.is_available():
#         print("CUDA is not available. Falling back to CPU. This will be very slow.")
#         device = "cpu"
#     else:
#         device = "cuda"
#         print("CUDA is available. Using GPU for acceleration.")

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, torch_dtype=torch.bfloat16, device_map=device
#     )
#     return model, tokenizer, device


def sanitize_tool(tool):
    """Removes the 'depends_on' key from a tool's dictionary."""
    tool_copy = tool.copy()
    if "depends_on" in tool_copy:
        del tool_copy["depends_on"]
    return tool_copy


# def query_llm_for_batch(prompt, model, tokenizer, device):
#     """Queries the LLM for a batch and returns the response text."""
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate a longer response to accommodate the JSON output
#     outputs = model.generate(
#         **inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id
#     )
#     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response_text[len(prompt) :].strip()


def parse_llm_response(response_text, batch_tool_names):
    """Parses the structured JSON output from the LLM to extract edges."""
    edges = set()
    try:
        # Find the JSON array within the response
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not json_match:
            # print(f"\n[Warning] No JSON array found in response: {response_text}")
            return edges

        connections = json.loads(json_match.group(0))

        for conn in connections:
            source = conn.get("source")
            target = conn.get("target")

            # Ensure both tools in the edge are actually from the current batch
            if (
                source in batch_tool_names
                and target in batch_tool_names
                and source != target
            ):
                edges.add(frozenset([source, target]))

    except json.JSONDecodeError:
        # Fallback for malformed JSON
        # print(f"\n[Warning] Could not decode JSON from LLM response: {response_text}")
        pass
    except Exception as e:
        # print(f"\n[Error] Error parsing LLM response: {e}")
        pass

    return edges


def create_hierarchical_batches(tools):
    """Creates batches of increasing size: 3, 6, 12, etc."""
    print("Creating hierarchical batches...")
    random.shuffle(tools)

    # Initial triplets
    num_tools = len(tools)
    triplets = [tools[i : i + 3] for i in range(0, num_tools, 3)]
    # Ensure all tools are included, even if the last group is smaller
    if len(triplets[-1]) < 3 and len(triplets) > 1:
        last_group = triplets.pop(-1)
        for i, item in enumerate(last_group):
            triplets[i % len(triplets)].append(item)

    all_levels = {3: triplets}
    current_level_batches = triplets
    current_size = 3

    while current_size * 2 < num_tools:
        next_size = current_size * 2
        next_level_batches = []

        random.shuffle(current_level_batches)  # Shuffle before pairing

        for i in range(0, len(current_level_batches), 2):
            if i + 1 < len(current_level_batches):
                # Merge two batches
                new_batch = current_level_batches[i] + current_level_batches[i + 1]
                next_level_batches.append(new_batch)
            else:
                # Add the leftover odd batch to the last created batch
                if next_level_batches:
                    next_level_batches[-1].extend(current_level_batches[i])
                else:  # Should not happen if there is more than one batch
                    next_level_batches.append(current_level_batches[i])

        all_levels[next_size] = next_level_batches
        current_level_batches = next_level_batches
        current_size = next_size
        print(
            f"  Created level with batch size ~{current_size} ({len(current_level_batches)} batches)"
        )

    return all_levels


def build_hierarchical_llm_graph():
    """Builds and evaluates the tool graph using hierarchical batching with an LLM."""

    print("--- Starting Hierarchical LLM Graph Builder ---")

    # --- 1. Load Checkpoint ---
    predicted_edges = set()
    last_processed_index = -1
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at {CHECKPOINT_PATH}. Loading progress...")
        with open(CHECKPOINT_PATH, "r") as f:
            try:
                checkpoint_data = json.load(f)
                last_processed_index = checkpoint_data.get("last_processed_index", -1)
                edges_list = checkpoint_data.get("predicted_edges", [])
                predicted_edges = {frozenset(edge) for edge in edges_list}
                print(
                    f"Resuming from batch index {last_processed_index + 1}. Found {len(predicted_edges)} edges so far."
                )
            except json.JSONDecodeError:
                print("Warning: Could not read checkpoint file. Starting from scratch.")

    # --- 2. Load Data & Ground Truth ---
    print("Loading toollinkos data...")
    with open("data/toollinkos/core_tools.json", "r") as f:
        core_tools = json.load(f)
    with open("data/toollinkos/regular_tools.json", "r") as f:
        regular_tools = json.load(f)
    all_tools = core_tools + regular_tools
    print(f"Loaded {len(all_tools)} tools.")

    ground_truth_edges = {
        frozenset([t["name"], dep["name"]])
        for t in all_tools
        for dep in t.get("depends_on", [])
    }
    print(f"Found {len(ground_truth_edges)} ground truth edges.")

    # --- 3. Create and Flatten Batches for easier checkpointing ---
    hierarchical_batches = create_hierarchical_batches(all_tools)
    all_batch_tasks = []
    for batch_size, batches in hierarchical_batches.items():
        for batch in batches:
            all_batch_tasks.append((batch_size, batch))

    # --- 4. Load LLM ---
    # model, tokenizer, device = load_llm()

    # --- 5. Define Prompt ---
    prompt_template = """Here are the types of dependencies to look for:
I. Tool Directly Depends On: A tool requires another to operate. (e.g., "set_wifi_on" before a tool needing internet).
II. Tool Indirectly Depends On: A tool benefits from another but doesn't strictly require it (e.g., "restaurant_reservation" might use "get_weather").
III. Parameter Directly Depends On: A required parameter must be obtained from another tool (e.g., "product_info" needs a "product_id").
IV. Parameter Indirectly Depends On: A parameter depends on context only if needed (e.g., "tomorrow" requires "get_current_date").

Analyze the list of tools provided below. Identify all pairs of tools (A, B) from this list where Tool A depends on Tool B, or Tool B depends on Tool A.

Return your findings as a single JSON array of objects, where each object represents a dependency. Do not include any other text or explanation.

Format: `[{{"source": "tool_name_1", "target": "tool_name_2"}}, {{"source": "tool_name_3", "target": "tool_name_4"}}]`

Tools:
{tool_batch}
"""

    # --- 6. Main Loop ---
    print(f"\nProcessing a total of {len(all_batch_tasks)} batches.")
    for i, (batch_size, batch) in enumerate(
        tqdm(all_batch_tasks, desc="Overall Progress")
    ):

        # Skip already processed batches if resuming
        if i <= last_processed_index:
            continue

        sanitized_batch = [sanitize_tool(tool) for tool in batch]
        batch_tool_names = {tool["name"] for tool in batch}
        # print(sanitized_batch)
        prompt = prompt_template.format(
            tool_batch=json.dumps(sanitized_batch, indent=2)
        )

        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly intelligent assistant that analyzes a list of software tools to map their dependencies.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        # print(response_text)
        # response_text = query_llm_for_batch(prompt, model, tokenizer, device)
        # torch.cuda.empty_cache()
        edges_from_batch = parse_llm_response(response_text, batch_tool_names)

        predicted_edges.update(edges_from_batch)
        if i % SAVE_EVERY == 0:
            curr_jaccard = jaccard_similarity(ground_truth_edges, predicted_edges)
            print(
                f"[Checkpoint] {len(predicted_edges)} edges — "
                f"Jaccard={curr_jaccard:.4f}",
                flush=True,
            )
            with open(CHECKPOINT_PATH, "w") as f:
                checkpoint_data = {
                    "last_processed_index": i,
                    "predicted_edges": [list(edge) for edge in predicted_edges],
                }
                json.dump(checkpoint_data, f)

    # --- 7. Evaluation and Results Saving ---
    jaccard = jaccard_similarity(ground_truth_edges, predicted_edges)

    print("\n--- Hierarchical LLM Evaluation Results ---")
    print(f"Ground Truth Edges: {len(ground_truth_edges)}")
    print(f"Predicted Edges (from LLM): {len(predicted_edges)}")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print("-----------------------------------------")

    # Save final results to a file
    final_results = {
        "ground_truth_edges_count": len(ground_truth_edges),
        "predicted_edges_count": len(predicted_edges),
        "jaccard_similarity": jaccard,
        "predicted_edges": [list(edge) for edge in predicted_edges],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to {RESULTS_PATH}")

    # --- 8. Cleanup ---
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint file removed.")


if __name__ == "__main__":
    build_hierarchical_llm_graph()
