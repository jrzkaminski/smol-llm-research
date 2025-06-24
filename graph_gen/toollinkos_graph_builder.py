import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import itertools


def normalize_text(text):
    """Normalize text for similarity comparison."""
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_type_category(type_str):
    """Categorize types into simple or complex."""
    simple_types = {'string', 'boolean', 'integer', 'number', 'float', 'bool'}
    if type_str in simple_types:
        return 'simple'
    return 'complex'


def can_types_match(type1, type2):
    """Check if two types can be matched according to the rules."""
    cat1 = get_type_category(type1)
    cat2 = get_type_category(type2)
    
    if cat1 == 'simple' and cat2 == 'simple':
        return type1 == type2
    elif cat1 == 'complex' or cat2 == 'complex':
        return True
    return False


def extract_parameters_info(parameters):
    """Extract type and description information from parameters."""
    info = []
    if not parameters:
        return info
    
    for param_info in parameters:
        param_type = param_info.get('type', 'object')
        param_desc = param_info.get('description', '')
        info.append({
            'name': param_info.get('name'),
            'type': param_type,
            'description': normalize_text(param_desc)
        })
    return info


def compute_cosine_similarity(texts1, texts2):
    """Compute cosine similarity between two sets of texts."""
    if not texts1 or not texts2:
        return 0.0
    
    all_texts = texts1 + texts2
    if len(set(all_texts)) <= 1:
        return 0.0
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        text1_indices = range(len(texts1))
        text2_indices = range(len(texts1), len(all_texts))
        
        sim_matrix = cosine_similarity(tfidf_matrix[text1_indices], tfidf_matrix[text2_indices])
        return np.max(sim_matrix) if sim_matrix.size > 0 else 0.0
    except:
        return 0.0


def normalize_similarity_matrix(matrix):
    """Normalize similarity matrix by dividing by the maximum value."""
    max_val = np.max(matrix)
    if max_val > 0:
        return matrix / max_val
    return matrix


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def build_toollinkos_graph():
    """Build and evaluate the toollinkos tool graph."""
    
    # Load data
    print("Loading toollinkos data...")
    with open('data/toollinkos/core_tools.json', 'r') as f:
        core_tools = json.load(f)
    with open('data/toollinkos/regular_tools.json', 'r') as f:
        regular_tools = json.load(f)
    
    all_tools = core_tools + regular_tools
    print(f"Loaded {len(all_tools)} tools in total.")

    # Extract ground truth graph
    ground_truth_edges = set()
    for tool in all_tools:
        tool_name = tool['name']
        for dependency in tool.get('depends_on', []):
            dep_name = dependency['name']
            # Undirected edge
            edge = frozenset([tool_name, dep_name])
            ground_truth_edges.add(edge)
    print(f"Found {len(ground_truth_edges)} ground truth edges.")

    # Build predicted graph
    predicted_edges = set()
    
    # Step 1 & 2: Type-based AND Description-based matching
    print("  Step 1 & 2: Type-based AND Description-based matching...")
    for tool1, tool2 in itertools.combinations(all_tools, 2):
        
        # NOTE: In toollinkos, tools don't have explicit "results". 
        # We'll assume a tool's output can be any of its parameter types.
        # This is a simplification. We'll connect tool1 -> tool2 if tool2's params match tool1's.
        
        tool1_params = extract_parameters_info(tool1.get('parameters', []))
        tool2_params = extract_parameters_info(tool2.get('parameters', []))
        
        # Since no explicit results, check params <-> params
        
        type_match = False
        for p1 in tool1_params:
            for p2 in tool2_params:
                if can_types_match(p1['type'], p2['type']):
                    type_match = True
                    break
            if type_match:
                break
        
        description_match = False
        if type_match and tool1_params and tool2_params:
            d1 = [p['description'] for p in tool1_params if p['description']]
            d2 = [p['description'] for p in tool2_params if p['description']]
            if d1 and d2:
                similarity = compute_cosine_similarity(d1, d2)
                if similarity > 0.9:
                    description_match = True
        
        if type_match and description_match:
            edge = frozenset([tool1['name'], tool2['name']])
            predicted_edges.add(edge)
            
    print(f"    Found {len(predicted_edges)} parameter-based connections.")

    # Step 3: Tool description similarity
    print("  Step 3: Tool description similarity...")
    tool_descriptions = [normalize_text(t.get('description', '')) for t in all_tools]
    tool_names = [t['name'] for t in all_tools]
    
    if len(tool_descriptions) > 1:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(tool_descriptions)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_matrix = normalize_similarity_matrix(similarity_matrix)
            
            for i in range(len(tool_names)):
                for j in range(i + 1, len(tool_names)): # Use i+1 to avoid self-loops and duplicates
                    if similarity_matrix[i, j] > 0.9:
                        edge = frozenset([tool_names[i], tool_names[j]])
                        predicted_edges.add(edge)
        except Exception as e:
            print(f"Error in tool description similarity: {e}")
            
    print(f"    Found {len(predicted_edges)} total predicted connections after step 3.")

    # Evaluate with Jaccard Similarity
    jaccard = jaccard_similarity(ground_truth_edges, predicted_edges)
    
    print("\n--- Evaluation Results ---")
    print(f"Ground Truth Edges: {len(ground_truth_edges)}")
    print(f"Predicted Edges: {len(predicted_edges)}")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    print("------------------------")


if __name__ == "__main__":
    build_toollinkos_graph() 