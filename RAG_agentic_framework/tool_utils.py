import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from RAG_agentic_framework.schemas import (
    GraphStructure,
    ToolSchema,
    ToolCall,
    BenchmarkItem,
    NestedObjectProperty,
    SimpleToolProperty,
    AnyToolProperty,
    ToolIOSchema,
)


def load_json(path: Path) -> Any:
    """Loads JSON data from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return None


def load_graph(path: Path) -> Optional[GraphStructure]:
    """Loads and validates graph structure."""
    data = load_json(path)
    if data:
        try:
            return GraphStructure.model_validate(data)
        except Exception as e:
            print(f"Error validating graph structure: {e}")
            return None
    return None


def load_tools(path: Path) -> Optional[Dict[str, ToolSchema]]:
    """Loads and validates tool descriptions, returning a dict keyed by tool name."""
    data = load_json(path)
    if data:
        try:
            tools_list = [ToolSchema.model_validate(item) for item in data]
            return {tool.name: tool for tool in tools_list}
        except Exception as e:
            print(f"Error validating tool descriptions: {e}")
            return None
    return None


def load_benchmark(path: Path) -> Optional[List[BenchmarkItem]]:  # Changed return type
    """Loads a list of benchmark items from a JSON file."""
    data = load_json(path)
    if data:
        if not isinstance(data, list):
            print(f"Error: Benchmark file at {path} does not contain a JSON list.")
            return None
        try:
            benchmark_items = [BenchmarkItem.model_validate(item) for item in data]
            print(f"Successfully loaded {len(benchmark_items)} benchmark items.")
            return benchmark_items
        except Exception as e:
            # Add more specific error handling if needed (e.g., which item failed)
            print(f"Error validating benchmark items: {e}")
            return None
    return None


def _format_properties(
    properties: Dict[str, AnyToolProperty],
    required: Optional[List[str]] = None,
    indent: str = "  ",
) -> str:
    """Recursively formats properties for display."""
    lines = []
    req_set = set(required or [])
    for name, prop in properties.items():
        req_marker = " (required)" if name in req_set else ""
        if isinstance(prop, SimpleToolProperty):
            lines.append(
                f"{indent}- {name}: {prop.type}{req_marker} ({prop.description or 'No description'})"
            )
        elif isinstance(prop, NestedObjectProperty):
            lines.append(
                f"{indent}- {name}: object{req_marker} ({prop.description or 'No description'}){{"
            )
            lines.append(
                _format_properties(prop.properties, prop.required, indent + "  ")
            )
            lines.append(f"{indent}}}")
        else:
            lines.append(f"{indent}- {name}: Unknown property type")
    return "\n".join(lines)


def get_tools_by_category(
    graph: GraphStructure, tools: Dict[str, ToolSchema]
) -> Dict[str, Dict[str, ToolSchema]]:
    """Groups tools by their category defined in the graph nodes."""
    tools_by_category: Dict[str, Dict[str, ToolSchema]] = {}
    if not graph or not isinstance(graph, GraphStructure):
        print("Warning: Invalid or missing graph structure.")
        return tools_by_category
    if not tools or not isinstance(tools, dict):
        print("Warning: Invalid or missing tool descriptions dictionary.")
        return tools_by_category

    categories = set()
    node_names_in_graph = set()

    for node in graph.nodes:
        if not hasattr(node, "name") or not hasattr(node, "category"):
            print(f"Warning: Skipping invalid node in graph.json: {node}")
            continue

        node_name = node.name
        node_category = node.category
        node_names_in_graph.add(node_name)
        categories.add(node_category)

        if node_name in tools:
            tool_schema = tools[node_name]
            if isinstance(tool_schema, ToolSchema):
                if node_category not in tools_by_category:
                    tools_by_category[node_category] = {}
                tools_by_category[node_category][node_name] = tool_schema
            else:
                print(
                    f"Warning: Item for tool '{node_name}' in loaded tools is not a valid ToolSchema object. Type: {type(tool_schema)}. Skipping for category '{node_category}'."
                )
        else:
            print(
                f"Warning: Tool '{node_name}' found in graph node but not found in the loaded tool descriptions (tools.json). Skipping this tool for category '{node_category}'."
            )

    print(f"\nFound categories in graph: {categories}")
    tool_names_in_schema = set(tools.keys())
    unassigned_tools = tool_names_in_schema - node_names_in_graph
    if unassigned_tools:
        print(
            f"Warning: The following tools exist in tools.json but are not assigned to a category in graph.json: {unassigned_tools}"
        )

    print(
        f"Tools grouped by category (final count): { {k: len(v) for k, v in tools_by_category.items()} }"
    )
    return tools_by_category


def validate_tool_call(
    tool_call: ToolCall, available_tools: Dict[str, ToolSchema]
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single tool call against the available tool schemas.
    Checks tool existence, top-level argument names, and required top-level args.
    Does NOT recursively validate nested structures by default.
    """
    tool_name = tool_call.tool
    params = tool_call.param

    if tool_name not in available_tools:
        return False, f"Tool '{tool_name}' does not exist or is not available."

    tool_schema = available_tools[tool_name]
    args_schema: Optional[ToolIOSchema] = tool_schema.arguments

    expected_top_level_arg_names = (
        set(args_schema.properties.keys()) if args_schema else set()
    )
    provided_arg_names = set(params.keys()) if params else set()

    extra_args = provided_arg_names - expected_top_level_arg_names
    if extra_args:
        return (
            False,
            f"Tool '{tool_name}' received unexpected top-level arguments: {', '.join(extra_args)}. Expected: {', '.join(expected_top_level_arg_names) or 'None'}.",
        )

    required_top_level_args = (
        set(args_schema.required) if args_schema and args_schema.required else set()
    )
    missing_args = required_top_level_args - provided_arg_names
    if missing_args:
        return (
            False,
            f"Tool '{tool_name}' is missing required top-level arguments: {', '.join(missing_args)}.",
        )
    return True, None


def validate_tool_call_sequence(
    proposed_calls: List[ToolCall], full_tools_schema: Dict[str, ToolSchema]
) -> Tuple[bool, Optional[str]]:
    """Validates a sequence of tool calls using the full schema."""
    if not isinstance(proposed_calls, list):
        return False, "Agent did not return a list of tool calls."

    for call_dict in proposed_calls:
        try:
            tool_call = ToolCall.model_validate(call_dict)
            is_valid, error = validate_tool_call(tool_call, full_tools_schema)
            if not is_valid:
                return False, error
        except Exception as e:
            return False, f"Invalid tool call structure: {call_dict}. Error: {e}"

    return True, None


def format_tool_descriptions(tools: Dict[str, ToolSchema]) -> str:
    """Formats tool descriptions for the agent prompt, using the nested schema."""
    if not tools:
        return "No tools available for this category."
    desc = []
    for name, schema in tools.items():

        args_str = "None"
        if schema.arguments and schema.arguments.properties:
            args_list = []
            props = schema.arguments.properties
            required = set(schema.arguments.required or [])
            for arg_name, arg_props in props.items():
                req_marker = " (required)" if arg_name in required else ""
                args_list.append(
                    f"{arg_name}: {arg_props.type}{req_marker} ({arg_props.description})"
                )
            args_str = ", ".join(args_list)
        print("Tool name:", name)
        print("Tool description:", schema.description)
        desc.append(f"- {name}: {schema.description}\n  Arguments: {args_str}")
    return "\n".join(desc)
