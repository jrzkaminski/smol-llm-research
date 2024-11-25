import json
import requests
import re
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any, Set
import logging

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

def extract_number_of_arguments(data: List[Dict[str, Any]]) -> List[int]:
    """
    Extract the number of arguments for each function in the dataset.

    Args:
        data (List[Dict[str, Any]]): The dataset loaded from the JSON file.

    Returns:
        List[int]: A list containing the number of arguments per function.
    """
    args_per_function = []

    for tool in data:
        function_descriptions = tool.get('Function_Description', {})
        for func_desc in function_descriptions.values():
            num_args = _parse_function_arguments(func_desc)
            args_per_function.append(num_args)
            logger.debug(f"Function '{func_desc[:30]}...' has {num_args} arguments")

    logger.info("Extracted number of arguments per function")
    return args_per_function

def _parse_function_arguments(func_desc: str) -> int:
    """
    Helper function to parse the number of arguments from a function description.

    Args:
        func_desc (str): The description of the function.

    Returns:
        int: The number of arguments found in the function.
    """
    try:
        match = re.search(r'Parameters:\s*(\{.*?\})\nOutput:', func_desc, re.DOTALL)
        if match:
            params_str = match.group(1)
            # Replace single quotes with double quotes for valid JSON
            params_str = params_str.replace("'", '"')
            params = json.loads(params_str)
            num_args = len(params)
            return num_args
        else:
            logger.warning(f"No parameters found in function description: {func_desc[:30]}...")
            return 0
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in function description: {e}")
        return 0

def extract_tools_per_task(data: List[Dict[str, Any]]) -> List[int]:
    """
    Calculate how many tools are needed to solve each task.

    Args:
        data (List[Dict[str, Any]]): The dataset loaded from the JSON file.

    Returns:
        List[int]: A list containing the number of tools used per task.
    """
    tools_per_task = []

    for tool in data:
        instances = tool.get('Instances', [])
        for instance in instances:
            tools_used = _extract_tools_from_instance(instance)
            num_tools = len(tools_used)
            tools_per_task.append(num_tools)
            logger.debug(f"Task uses {num_tools} tools: {tools_used}")

    logger.info("Extracted number of tools used per task")
    return tools_per_task

def _extract_tools_from_instance(instance: Dict[str, Any]) -> Set[str]:
    """
    Helper function to extract the set of tools used in an instance.

    Args:
        instance (Dict[str, Any]): An instance from the dataset.

    Returns:
        Set[str]: A set of tool names used in the instance.
    """
    tools_used = set()
    intermediate_steps = instance.get('intermediate_steps', [])

    for step in intermediate_steps:
        if step and isinstance(step, list) and step[0]:
            action = step[0][0]
            tools_used.add(action)
            logger.debug(f"Found tool action: {action}")

    return tools_used

def extract_functions_per_api(data: List[Dict[str, Any]]) -> List[int]:
    """
    Extract the number of functions each API provides.

    Args:
        data (List[Dict[str, Any]]): The dataset loaded from the JSON file.

    Returns:
        List[int]: A list containing the number of functions per API.
    """
    functions_per_api = []

    for tool in data:
        function_descriptions = tool.get('Function_Description', {})
        num_functions = len(function_descriptions)
        functions_per_api.append(num_functions)
        logger.debug(f"API '{tool.get('Name')}' has {num_functions} functions")

    logger.info("Extracted number of functions per API")
    return functions_per_api

def extract_task_characteristics(data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Extract characteristics of agent task solves, such as number of steps and tools used.

    Args:
        data (List[Dict[str, Any]]): The dataset loaded from the JSON file.

    Returns:
        Dict[str, List[int]]: A dictionary containing lists of characteristics.
    """
    num_steps_per_task = []
    tools_used_per_task = []

    for tool in data:
        instances = tool.get('Instances', [])
        for instance in instances:
            intermediate_steps = instance.get('intermediate_steps', [])
            num_steps = len(intermediate_steps)
            num_steps_per_task.append(num_steps)

            tools_used = _extract_tools_from_instance(instance)
            tools_used_per_task.append(len(tools_used))

            logger.debug(f"Task has {num_steps} steps and uses {len(tools_used)} tools")

    logger.info("Extracted task characteristics")
    return {
        'num_steps_per_task': num_steps_per_task,
        'tools_used_per_task': tools_used_per_task
    }

def plot_distribution(data: List[int], title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """
    Plot and save a histogram of the provided data.

    Args:
        data (List[int]): The data to plot.
        title (str): The title of the plot.
        xlabel (str): The X-axis label.
        ylabel (str): The Y-axis label.
        filename (str): The filename to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(1, max(data) + 2), edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(1, max(data) + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Plot saved as {filename}")

def main():
    """
    Main function to execute data processing and visualization.
    """
    DATA_URL = 'https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/refs/heads/main/data/train_data.json'

    # Load the dataset
    dataset = load_dataset(DATA_URL)

    # 1. Distribution of Number of Arguments per Tool Function
    args_per_function = extract_number_of_arguments(dataset)
    plot_distribution(
        data=args_per_function,
        title='Distribution of Number of Arguments per Function',
        xlabel='Number of Arguments',
        ylabel='Frequency',
        filename='args_per_function.png'
    )

    # Distribution of Number of Tools Used per Task
    tools_per_task = extract_tools_per_task(dataset)
    plot_distribution(
        data=tools_per_task,
        title='Distribution of Number of Tools Used per Task',
        xlabel='Number of Tools',
        ylabel='Frequency',
        filename='tools_per_task.png'
    )

    # 2. Distribution of Number of Functions per API
    functions_per_api = extract_functions_per_api(dataset)
    plot_distribution(
        data=functions_per_api,
        title='Distribution of Number of Functions per API',
        xlabel='Number of Functions',
        ylabel='Frequency',
        filename='functions_per_api.png'
    )

    # 3. Characteristics of Agent Task Solves
    task_characteristics = extract_task_characteristics(dataset)

    # Plot Number of Intermediate Steps per Task
    plot_distribution(
        data=task_characteristics['num_steps_per_task'],
        title='Distribution of Number of Intermediate Steps per Task',
        xlabel='Number of Steps',
        ylabel='Frequency',
        filename='num_steps_per_task.png'
    )

    # Plot Number of Tools Used per Task
    plot_distribution(
        data=task_characteristics['tools_used_per_task'],
        title='Distribution of Number of Tools Used per Task',
        xlabel='Number of Tools',
        ylabel='Frequency',
        filename='tools_used_per_task.png'
    )

    # Additional Analysis: Most Frequently Used Tools
    tool_usage_counts = count_tool_usage(dataset)
    plot_most_common_tools(tool_usage_counts, top_n=10, filename='most_common_tools.png')

    # Print Average Number of Arguments per Function
    average_args = sum(args_per_function) / len(args_per_function)
    logger.info(f'Average Number of Arguments per Function: {average_args:.2f}')

def count_tool_usage(data: List[Dict[str, Any]]) -> Counter:
    """
    Count the usage frequency of each tool in the dataset.

    Args:
        data (List[Dict[str, Any]]): The dataset loaded from the JSON file.

    Returns:
        Counter: A Counter object mapping tools to their usage counts.
    """
    tool_usage_counter = Counter()

    for tool in data:
        instances = tool.get('Instances', [])
        for instance in instances:
            tools_used = _extract_tools_from_instance(instance)
            tool_usage_counter.update(tools_used)

    logger.info("Counted tool usage frequency")
    return tool_usage_counter

def plot_most_common_tools(tool_usage_counts: Counter, top_n: int, filename: str) -> None:
    """
    Plot the most frequently used tools.

    Args:
        tool_usage_counts (Counter): A Counter object with tool usage counts.
        top_n (int): The number of top tools to display.
        filename (str): The filename to save the plot.
    """
    most_common_tools = tool_usage_counts.most_common(top_n)
    tools, counts = zip(*most_common_tools)

    plt.figure(figsize=(10, 6))
    plt.barh(tools, counts, color='skyblue')
    plt.title(f'Top {top_n} Most Frequently Used Tools')
    plt.xlabel('Usage Count')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Most common tools plot saved as {filename}")

if __name__ == '__main__':
    main()
