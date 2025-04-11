import json
import requests
import re
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Any, Set
import logging
import itertools
import matplotlib.colors as mcolors

# Set up logging for debugging and maintenance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(url: str) -> List[Dict[str, Any]]:
    """
    Load the dataset from the provided URL.

    Args:
        url (str): The URL of the JSON dataset.

    Returns:
        List[Dict[str, Any]]: The list of data entries.
    """
    try:
        logger.info(f"Loading dataset from {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = json.loads(response.text)
        logger.info("Dataset loaded successfully")
        return data
    except requests.RequestException as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise

def extract_api_functions(data: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Extract functions for each API.

    Args:
        data (List[Dict[str, Any]]): The dataset.

    Returns:
        Dict[str, Set[str]]: A mapping from API name to its set of functions.
    """
    api_functions = {}

    for tool in data:
        api_name = tool.get('Name')
        functions = set(tool.get('Function_Description', {}).keys())
        api_functions[api_name] = functions
        logger.debug(f"API '{api_name}' functions: {functions}")

    logger.info("Extracted functions for each API")
    return api_functions

def extract_api_paths(data: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    """
    Extract intermediate steps (paths) for each API.

    Args:
        data (List[Dict[str, Any]]): The dataset.

    Returns:
        Dict[str, List[List[str]]]: A mapping from API name to a list of function call sequences.
    """
    api_paths = defaultdict(list)

    for tool in data:
        api_name = tool.get('Name')
        instances = tool.get('Instances', [])

        for instance in instances:
            path = []
            intermediate_steps = instance.get('intermediate_steps', [])

            for step in intermediate_steps:
                if step and isinstance(step, list) and step[0]:
                    action = step[0][0]
                    path.append(action)
                    logger.debug(f"Instance step action: {action}")

            if path:
                api_paths[api_name].append(path)
                logger.debug(f"API '{api_name}' path: {path}")

    logger.info("Extracted paths for each API")
    return api_paths

def build_api_graphs(api_functions: Dict[str, Set[str]], api_paths: Dict[str, List[List[str]]]) -> Dict[str, nx.MultiDiGraph]:
    """
    Build graphs for each API.

    Args:
        api_functions (Dict[str, Set[str]]): API functions.
        api_paths (Dict[str, List[List[str]]]): API paths.

    Returns:
        Dict[str, nx.MultiDiGraph]: A mapping from API name to its graph.
    """
    api_graphs = {}

    for api_name, functions in api_functions.items():
        G = nx.MultiDiGraph()
        G.add_nodes_from(functions)
        logger.debug(f"Created graph for API '{api_name}' with nodes: {functions}")

        paths = api_paths.get(api_name, [])
        for idx, path in enumerate(paths):
            # Add edges for each path with an identifier
            edges = list(zip(path[:-1], path[1:]))
            for edge in edges:
                G.add_edge(edge[0], edge[1], key=idx)
                logger.debug(f"API '{api_name}' added edge {edge} for path {idx}")

        api_graphs[api_name] = G

    logger.info("Built graphs for each API")
    return api_graphs

def visualize_api_graph(api_name: str, G: nx.MultiDiGraph, paths: List[List[str]], output_dir: str = 'graphs') -> None:
    """
    Visualize the API graph with paths.

    Args:
        api_name (str): The name of the API.
        G (nx.MultiDiGraph): The graph of the API.
        paths (List[List[str]]): The list of function call sequences.
        output_dir (str): The directory to save the graphs.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 8))

    # Generate a color palette
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_cycle = itertools.cycle(colors)

    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with different colors for each path
    for idx, path in enumerate(paths):
        edge_list = list(zip(path[:-1], path[1:]))
        edge_color = next(color_cycle)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_list,
            width=2,
            edge_color=edge_color,
            arrows=True,
            arrowstyle='->',
            arrowsize=15,
            connectionstyle='arc3,rad=0.1'  # Slight curve to distinguish overlapping edges
        )
        logger.debug(f"Drew edges for path {idx} with color {edge_color}")

    plt.title(f"API Function Graph for '{api_name}'")
    plt.axis('off')
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{api_name.replace('/', '_')}_graph.png")
    plt.savefig(filename)
    plt.close()
    logger.info(f"Graph for API '{api_name}' saved as {filename}")

def main():
    """
    Main function to execute data processing and visualization.
    """
    DATA_URL = 'https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/refs/heads/main/data/train_data.json'

    # Load the dataset
    dataset = load_dataset(DATA_URL)

    # Extract functions and paths for each API
    api_functions = extract_api_functions(dataset)
    api_paths = extract_api_paths(dataset)

    # Build graphs for each API
    api_graphs = build_api_graphs(api_functions, api_paths)

    # Visualize each API graph
    for api_name, G in api_graphs.items():
        paths = api_paths.get(api_name, [])
        if paths:
            visualize_api_graph(api_name, G, paths)
        else:
            logger.warning(f"No paths found for API '{api_name}', skipping visualization.")

if __name__ == '__main__':
    main()
