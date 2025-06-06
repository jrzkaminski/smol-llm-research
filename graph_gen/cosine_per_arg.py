import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re


def normalize_text(text):
    """Normalize text for similarity comparison."""
    if not text:
        return ""
    # Convert to lowercase, remove special characters, normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_type_category(type_str):
    """Categorize types into simple or complex."""
    simple_types = {'string', 'boolean', 'integer', 'number', 'float'}
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


def extract_properties_info(properties):
    """Extract type and description information from properties."""
    info = []
    if not properties:
        return info
    
    for prop_name, prop_info in properties.items():
        prop_type = prop_info.get('type', 'object')
        prop_desc = prop_info.get('description', '')
        info.append({
            'name': prop_name,
            'type': prop_type,
            'description': normalize_text(prop_desc)
        })
    return info


def compute_cosine_similarity(texts1, texts2):
    """Compute cosine similarity between two sets of texts."""
    if not texts1 or not texts2:
        return 0.0
    
    all_texts = texts1 + texts2
    if len(set(all_texts)) <= 1:  # All texts are the same or empty
        return 0.0
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities between texts1 and texts2
        similarities = []
        for i in range(len(texts1)):
            for j in range(len(texts1), len(all_texts)):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0, 0]
                similarities.append(sim)
        
        return max(similarities) if similarities else 0.0
    except:
        return 0.0


def normalize_similarity_matrix(matrix):
    """Normalize similarity matrix by dividing by the maximum value."""
    max_val = np.max(matrix)
    if max_val > 0:
        return matrix / max_val
    return matrix


def build_tool_graph():
    """Build the tool graph with the specified steps."""
    
    # Load data
    print("Loading data...")
    with open('data/ultratool/tools.json', 'r') as f:
        tools_data = json.load(f)
    
    with open('data/ultratool/graph.json', 'r') as f:
        graph_data = json.load(f)
    
    # Create a mapping from tool name to category
    tool_to_category = {}
    for node in graph_data['nodes']:
        tool_to_category[node['name']] = node['category']
    
    # Organize tools by category
    tools_by_category = defaultdict(list)
    for tool in tools_data:
        category = tool_to_category.get(tool['name'])
        if category:
            tools_by_category[category].append(tool)
    
    print(f"Found {len(tools_by_category)} categories")
    
    # Final graph structure
    final_edges = []
    
    # Process each category
    for category, tools in tools_by_category.items():
        print(f"\nProcessing category: {category} ({len(tools)} tools)")
        
        # Step 1 & 2 Combined: Type-based AND Description-based matching
        print("  Step 1 & 2: Type-based AND Description-based matching...")
        combined_edges = []
        for i, tool1 in enumerate(tools):
            for j, tool2 in enumerate(tools):
                if i == j:
                    continue
                
                # Extract argument and result information
                tool1_args = extract_properties_info(tool1.get('arguments', {}).get('properties', {}))
                tool1_results = extract_properties_info(tool1.get('results', {}).get('properties', {}))
                tool2_args = extract_properties_info(tool2.get('arguments', {}).get('properties', {}))
                tool2_results = extract_properties_info(tool2.get('results', {}).get('properties', {}))
                
                # Check if types match AND descriptions are similar
                type_match = False
                description_match = False
                
                # Check type compatibility
                for result in tool1_results:
                    for arg in tool2_args:
                        if can_types_match(result['type'], arg['type']):
                            type_match = True
                            break
                    if type_match:
                        break
                
                # Check description similarity if types match
                if type_match and tool1_results and tool2_args:
                    result_descriptions = [r['description'] for r in tool1_results if r['description']]
                    arg_descriptions = [a['description'] for a in tool2_args if a['description']]
                    
                    if result_descriptions and arg_descriptions:
                        similarity = compute_cosine_similarity(result_descriptions, arg_descriptions)
                        if similarity > 0.3:
                            description_match = True
                
                # Only add edge if BOTH type and description match
                if type_match and description_match:
                    edge = (tool1['name'], tool2['name'])
                    if edge not in combined_edges:
                        combined_edges.append(edge)
        
        print(f"    Found {len(combined_edges)} type+description-based connections")
        
        # Step 3: Tool description similarity
        print("  Step 3: Tool description similarity...")
        tool_desc_edges = []
        tool_descriptions = []
        tool_names = []
        
        for tool in tools:
            desc = normalize_text(tool.get('description', ''))
            if desc:
                tool_descriptions.append(desc)
                tool_names.append(tool['name'])
        
        if len(tool_descriptions) > 1:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(tool_descriptions)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Normalize the similarity matrix
                similarity_matrix = normalize_similarity_matrix(similarity_matrix)
                
                for i in range(len(tool_names)):
                    for j in range(len(tool_names)):
                        if i != j and similarity_matrix[i, j] > 0.5:
                            edge = (tool_names[i], tool_names[j])
                            if edge not in tool_desc_edges:
                                tool_desc_edges.append(edge)
            except:
                pass
        
        print(f"    Found {len(tool_desc_edges)} tool description-based connections")
        
        # Combine all edges for this category
        all_category_edges = set(combined_edges + tool_desc_edges)
        final_edges.extend(list(all_category_edges))
        print(f"  Total connections for {category}: {len(all_category_edges)}")
    
    # Create the final graph structure
    print(f"\nTotal edges across all categories: {len(final_edges)}")
    
    # Convert edges to the format expected by graph.json
    links = []
    for source, target in final_edges:
        links.append({
            "source": source,
            "target": target
        })
    
    # Create the new graph
    new_graph = {
        "links": links,
        "nodes": graph_data['nodes']  # Keep the same nodes
    }
    
    # Save the new graph
    output_file = 'data/ultratool/new_graph.json'
    with open(output_file, 'w') as f:
        json.dump(new_graph, f, indent=2)
    
    print(f"\nNew graph saved to {output_file}")
    print(f"Total nodes: {len(new_graph['nodes'])}")
    print(f"Total links: {len(new_graph['links'])}")
    
    # Compare with original graph
    original_links = len(graph_data['links'])
    new_links = len(new_graph['links'])
    print(f"\nComparison with original graph:")
    print(f"  Original graph links: {original_links}")
    print(f"  New graph links: {new_links}")
    print(f"  Difference: {new_links - original_links} ({((new_links - original_links) / original_links * 100):+.1f}%)")
    
    # Print some statistics
    print("\nStatistics by category:")
    links_by_category = defaultdict(int)
    for link in links:
        source_cat = tool_to_category.get(link['source'], 'Unknown')
        links_by_category[source_cat] += 1
    
    for category, count in sorted(links_by_category.items()):
        print(f"  {category}: {count} outgoing connections")


if __name__ == "__main__":
    build_tool_graph()
