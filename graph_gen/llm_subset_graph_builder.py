import json
import itertools
import random
import os
import re
import warnings
from typing import List, Dict, Tuple

import torch
import torch._dynamo
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# ----------------------- CONFIGURATION -----------------------
MODEL_ID_LLM = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
MODEL_FILE = "DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PROGRESS_PATH = "progress_edges_subset.json"
SIMILARITY_CACHE = "tool_description_embeddings.pt"
SAVE_EVERY = 500               # checkpoint frequency (iterations)
N_ITERATIONS = 800_0         # total LLM calls
SUBSET_SIZE_RANGE = (3, 10)    # inclusive (min, max)
SIMILARITY_PROB = 0.5          # probability of similarity-based sampling
TEMPERATURE = 0.6              # DeepSeek-R1 recommended temperature
# -------------------------------------------------------------

torch._dynamo.config.cache_size_limit = 64
warnings.filterwarnings(
    "ignore", message="Passing `pad_token_id` to `eos_token_id`")

def load_tools() -> List[Dict]:
    """Load core and regular tools as a single list."""
    with open("data/toollinkos/core_tools.json", "r") as f:
        core_tools = json.load(f)
    with open("data/toollinkos/regular_tools.json", "r") as f:
        regular_tools = json.load(f)
    return core_tools + regular_tools

def build_ground_truth_edges(tools: List[Dict]) -> set:
    """Return undirected edges present in the ground-truth depends_on field."""
    edges = set()
    for t in tools:
        for dep in t.get("depends_on", []):
            edges.add(frozenset([t["name"], dep["name"]]))
    return edges

# ----------------------- LLM HELPERS -------------------------

def load_llm(model_id: str = MODEL_ID_LLM, model_file: str = MODEL_FILE):
    print("Loading GGUF model… (may take a while the first time)")
    
    # Check if llama-cpp-python has CUDA support
    try:
        from llama_cpp import llama_cpp
        has_cuda = hasattr(llama_cpp, 'GGML_USE_CUDA') or hasattr(llama_cpp, 'LLAMA_SUPPORTS_GPU_OFFLOAD')
        print(f"llama-cpp-python CUDA support: {has_cuda}")
    except:
        print("Could not check CUDA support in llama-cpp-python")
    
    # Configure GPU usage
    if torch.cuda.is_available():
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA detected – GPU memory: {gpu_memory_gb:.1f}GB")
        
        # Check current GPU memory usage
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        cached_gb = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU memory before loading: {allocated_gb:.2f}GB allocated, {cached_gb:.2f}GB cached")
        
            
        # Force GPU memory allocation
        torch.cuda.empty_cache()
    else:
        n_gpu_layers = 0
        print("CUDA not available – falling back to CPU (slow).")
    
    # Load the GGUF model using llama-cpp-python
    model = Llama.from_pretrained(
        repo_id=model_id,
        filename=model_file,
        n_gpu_layers=-1,
        n_ctx=4096,  # Context length
        verbose=True,  # Enable verbose to see GPU usage
        use_mmap=True,  # Memory mapping for efficiency
        use_mlock=False,  # Don't lock memory pages
    )
    
    # Check GPU memory usage after loading
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        cached_gb = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU memory after loading: {allocated_gb:.2f}GB allocated, {cached_gb:.2f}GB cached")
        
        # Try a simple test to verify GPU usage
        print("Testing GPU usage with a simple prompt...")
        test_response = model("Test", max_tokens=5, temperature=0.1)
        allocated_gb_after = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"GPU memory after inference: {allocated_gb_after:.2f}GB allocated")
    
    return model

def sanitize_tool(tool: Dict) -> Dict:
    """Remove keys that leak ground truth (e.g., depends_on)."""
    t = tool.copy()
    t.pop("depends_on", None)
    return t

def query_llm(prompt: str, model) -> str:
    """Return raw text response from LLM using llama-cpp-python."""
    # Format prompt with DeepSeek-R1 chat template
    formatted_prompt = f"<｜User｜>{prompt}<｜Assistant｜>"
    
    # Generate response with thinking encouraged
    response = model(
        formatted_prompt,
        max_tokens=2048,
        temperature=TEMPERATURE,
        top_p=0.95,
        stop=["<｜User｜>"],
        echo=False
    )
    
    return response["choices"][0]["text"].strip()

# ----------------------- EMBEDDINGS --------------------------

def load_or_compute_embeddings(tools: List[Dict]) -> torch.Tensor:
    """Compute or load cached sentence embeddings for each tool description."""
    if os.path.isfile(SIMILARITY_CACHE):
        print("Loading cached tool embeddings…")
        return torch.load(SIMILARITY_CACHE)

    print("Computing tool description embeddings…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = SentenceTransformer(EMBED_MODEL_ID, device=device)

    descriptions = []
    for t in tools:
        desc = t.get("description", "")
        # include parameter names in description for more context
        if "parameters" in t and isinstance(t["parameters"], dict):
            desc += " " + " ".join(t["parameters"].keys())
        descriptions.append(desc)

    emb = emb_model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    torch.save(emb, SIMILARITY_CACHE)
    return emb

def cosine_similarity_matrix(emb: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity matrix (n_tools x n_tools)."""
    emb_norm = emb / emb.norm(dim=1, keepdim=True)
    return emb_norm @ emb_norm.T

# ----------------------- SAMPLING ----------------------------

def sample_subset_random(n_tools: int, k: int) -> List[int]:
    """Return a list of k unique indices chosen uniformly at random."""
    return random.sample(range(n_tools), k)

def sample_subset_similar(sim_mat: torch.Tensor, k: int) -> List[int]:
    """Select a subset of size k with high internal similarity.

    Strategy: pick a random seed index, then greedily choose k-1 highest-similarity
    indices to the seed (excluding itself)."""
    n_tools = sim_mat.shape[0]
    seed = random.randrange(n_tools)
    sim_scores = sim_mat[seed].cpu().numpy()
    sorted_idx = sim_scores.argsort()[::-1]  # descending
    subset = [seed]
    for idx in sorted_idx:
        if idx != seed and len(subset) < k:
            subset.append(int(idx))
        if len(subset) == k:
            break
    # Fallback to random fill (should rarely happen)
    if len(subset) < k:
        remaining = [i for i in range(n_tools) if i not in subset]
        subset.extend(random.sample(remaining, k - len(subset)))
    return subset

# ----------------------- PARSING -----------------------------

def extract_edges_from_response(resp: str) -> List[Tuple[str, str]]:
    """Parse JSON array of {source,target} objects anywhere in the response."""
    try:
        # Look for JSON after any thinking tags
        text = resp
        if "<think>" in text and "</think>" in text:
            # Extract content after thinking
            text = text.split("</think>")[-1]
        
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return []
        data = json.loads(match.group(0))
        edges = []
        for item in data:
            if isinstance(item, dict) and "source" in item and "target" in item:
                edges.append((item["source"], item["target"]))
        return edges
    except json.JSONDecodeError:
        return []

# --------------------- JACCARD -------------------------------

def jaccard_similarity(set1: set, set2: set) -> float:
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0

# ---------------------- MAIN LOOP ----------------------------

def main():
    random.seed(42)

    # 1. Load tools & ground truth
    tools = load_tools()
    ground_truth_edges = build_ground_truth_edges(tools)
    print(f"Loaded {len(tools)} tools. Ground truth edges: {len(ground_truth_edges)}")

    # 2. Embeddings & similarity matrix
    embeddings = load_or_compute_embeddings(tools)
    sim_mat = cosine_similarity_matrix(embeddings)

    # 3. Load / init progress
    if os.path.isfile(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            prog = json.load(f)
        
        raw_edges = prog.get("edges", [])
        # Filter out self-loops which would create frozensets of size 1 and cause errors
        predicted_edges = {
            frozenset([e["source"], e["target"]]) 
            for e in raw_edges 
            if e.get("source") != e.get("target") and e.get("source") and e.get("target")
        }

        jaccard_hist = prog.get("jaccard_history", [])
        start_iter = len(jaccard_hist)
        print(f"Resuming from checkpoint – {start_iter} iterations done, {len(predicted_edges)} predicted edges.")
    else:
        predicted_edges = set()
        jaccard_hist = []
        start_iter = 0

    # 4. Load LLM
    model = load_llm()

    # 5. Prompt template (subset-based with thinking encouragement)
    prompt_template = """
You are a highly intelligent assistant that analyzes software tools to determine if they depend on each other.

Please think through this step by step, considering the functionality and requirements of each tool.

Dependency types:
I. Tool Directly Depends On: A tool requires another tool to operate.
II. Tool Indirectly Depends On: A tool benefits from another but does not strictly require it.
III. Parameter Directly Depends On: A required parameter must be obtained from another tool.
IV. Parameter Indirectly Depends On: A parameter depends on additional context only if required by the user.

Given the following list of tools, identify ANY pairs of tools that exhibit ANY of the dependency types I-IV. Think carefully about each tool's purpose, parameters, and how they might relate to other tools.

After your analysis, return ONLY a JSON array where each element is an object with keys "source" and "target", representing an undirected dependency between the two tools. If there are no dependencies, return an empty array []. Do not output anything else after the JSON.

TOOLS:
{tools_json}
"""

    # 6. Iterative process
    progress_edges_json = [
        {"source": list(e)[0], "target": list(e)[1]} for e in predicted_edges
    ]

    for itr in tqdm(range(start_iter, N_ITERATIONS), desc="LLM iterations"):
        k = random.randint(*SUBSET_SIZE_RANGE)
        if random.random() < SIMILARITY_PROB:
            idx_subset = sample_subset_similar(sim_mat, k)
        else:
            idx_subset = sample_subset_random(len(tools), k)

        subset_tools = [sanitize_tool(tools[i]) for i in idx_subset]
        tools_json_str = json.dumps(subset_tools, indent=2)
        prompt = prompt_template.format(tools_json=tools_json_str)

        try:
            response = query_llm(prompt, model)
            new_edges = extract_edges_from_response(response)

            for src, tgt in new_edges:
                # Ensure src and tgt are strings, not lists
                if isinstance(src, list):
                    src = str(src)
                if isinstance(tgt, list):
                    tgt = str(tgt)
                
                # Skip self-loops
                if src == tgt:
                    continue
                    
                edge = frozenset([src, tgt])
                if edge not in predicted_edges:
                    predicted_edges.add(edge)
                    progress_edges_json.append({"source": src, "target": tgt})
        except Exception as e:
            tqdm.write(f"[Warning] Error in iteration {itr+1}: {str(e)} - Skipping this iteration")
            continue

        # --- metrics & checkpoints ---
        jac = jaccard_similarity(ground_truth_edges, predicted_edges)
        jaccard_hist.append(jac)

        if (itr + 1) % SAVE_EVERY == 0:
            with open(PROGRESS_PATH, "w") as f:
                json.dump({
                    "edges": progress_edges_json,
                    "jaccard_history": jaccard_hist
                }, f, indent=2)
            tqdm.write(f"[Checkpoint] Iter {itr+1} – Jaccard={jac:.4f}, total_edges={len(predicted_edges)}")

    # 7. Final save
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "edges": progress_edges_json,
            "jaccard_history": jaccard_hist
        }, f, indent=2)

    # 8. Plot
    plt.figure(figsize=(8, 4))
    plt.plot(jaccard_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Jaccard similarity")
    plt.title("Jaccard similarity over time (subset-based search)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("jaccard_over_time.png")
    plt.show()


if __name__ == "__main__":
    main() 