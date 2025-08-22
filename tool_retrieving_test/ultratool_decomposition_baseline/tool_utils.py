import json
from pathlib import Path
from typing import Optional, Any

from schemas import ToolSchema, BenchmarkItem


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


def load_tools(path: Path) -> Optional[dict[str, ToolSchema]]:
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


def load_benchmark(path: Path) -> Optional[list[BenchmarkItem]]:  # Changed return type
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


def simple_format_tool_descriptions(tools: dict[str, ToolSchema]) -> str:
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
        desc.append(f"- {name}: {schema.description}\nArguments: {args_str}")
    return "\n".join(desc)
