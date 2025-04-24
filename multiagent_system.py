from typing import TypedDict, Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from config import VSEGPT_KEY
import json
import warnings

warnings.filterwarnings("ignore")

llm = ChatOpenAI(
    model_name="meta-llama/llama-3.3-70b-instruct",
    openai_api_key=VSEGPT_KEY,
    openai_api_base="https://api.vsegpt.ru/v1",
)


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    graph_output: str


class OverallState(TypedDict):
    user_input: str
    graph_output: str
    reasoning_chain: List[str]


class ToolMetadata(TypedDict):
    id: str
    desc: str
    category: str


class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, ToolMetadata] = {}
        self.edges: List[Tuple[str, str]] = []
        self.START = "START"
        self.END = "END"

    def add_node(self, metadata: ToolMetadata) -> None:
        self.nodes[metadata["id"]] = metadata

    def add_edge(self, from_node: str, to_node: Optional[str]) -> None:
        if from_node and to_node:
            self.edges.append((from_node, to_node))


tools_json = json.load(open("ultratool_dev_graph.json", "r"))


graphs_by_category: Dict[str, StateGraph] = {}

for tool in tools_json["nodes"]:
    category = tool["category"]
    if category not in graphs_by_category:
        graphs_by_category[category] = StateGraph()
    graphs_by_category[category].add_node(tool)

for link in tools_json["links"]:
    source_tool = next(
        (t for t in tools_json["nodes"] if t["id"] == link["source"]), None
    )
    target_tool = next(
        (t for t in tools_json["nodes"] if t["id"] == link["target"]), None
    )
    if (
        source_tool
        and target_tool
        and source_tool["category"] == target_tool["category"]
    ):
        category = source_tool["category"]
        graphs_by_category[category].add_edge(link["source"], link["target"])


def supervisor_router(state: OverallState) -> str:
    available_categories = list(graphs_by_category.keys())
    categories_str = ", ".join(available_categories)
    prompt = (
        "You are a routing supervisor responsible for directing user requests to the appropriate agent-worker. "
        f"The available categories are: [{categories_str}].\n\n"
        f"User's request:\n{ state["user_input"]}\n\n"
        "Based on the request, decide which single category is the most relevant. "
        "Return the category name exactly as it appears in the provided list. Without special characters"
        "example: ['tool1', 'tool2']"
    )
    response = llm(prompt).content.strip()
    state["reasoning_chain"].append(
        f"Supervisor Router Prompt: {prompt}\nResponse: {response}"
    )
    # print(response)
    # print()
    return response


final_ans = ""


def agent_worker(
    state: OverallState, category: str, feedback: Optional[str] = None
) -> OverallState:
    if category not in graphs_by_category:
        state[
            "graph_output"
        ] += f"\n[Error] No available tools for category: {category}"
        return state

    graph = graphs_by_category[category]

    tools_list_lines = []
    for tool_id, metadata in graph.nodes.items():
        tools_list_lines.append(f"{tool_id}: {metadata['desc']}")
    tools_list_str = "\n".join(tools_list_lines)

    edges_list_lines = []
    for source, target in graph.edges:
        edges_list_lines.append(f"{source} -> {target}")
    edges_list_str = "\n".join(edges_list_lines)

    prompt = (
        f"You are an agent-worker specialized in the '{category}' category. "
        "Your task is to propose an optimal sequence of tool calls to address the following user request.\n\n"
        f"User's request:\n{state["user_input"]}\n\n"
        f"Available tools:\n{tools_list_str}\n\n"
        f"Tool connections:\n{edges_list_str}\n\n"
    )
    if feedback:
        prompt += (
            "The supervisor provided the following feedback on your previous attempt:\n"
            f"{feedback}\n\n"
            "Please revise your tool sequence accordingly.\n"
        )
    prompt += (
        "Return the sequence of tool identifiers as a Python list along with a brief explanation for each step."
        "Without your comments. Only the list of tools."
    )

    response = llm(prompt).content.strip()
    state["graph_output"] += f"\n[Agent-Worker Output for {category}]\n{response}"
    state["reasoning_chain"].append(
        f"Agent-Worker Prompt for {category}: {prompt}\nResponse: {response}"
    )
    global final_ans
    final_ans = response
    return state


def supervisor_evaluate(state: OverallState) -> Tuple[bool, str]:
    prompt = (
        "You are a supervisor agent. Please review the following proposed tool sequence along with its explanations "
        "provided by the agent-worker:\n\n"
        f"{state['graph_output']}\n\n"
        "Evaluate whether this sequence correctly and optimally addresses the user's request. "
        "If it is acceptable, reply with exactly 'APPROVED'. If modifications are needed, reply with 'REWORK:' "
        "followed by specific feedback on what should be improved."
    )
    response = llm(prompt).content.strip()
    state["reasoning_chain"].append(
        f"Supervisor Evaluation Prompt: {prompt}\nResponse: {response}"
    )
    # print(response)
    # print()
    if "APPROVED" in response.upper():
        return True, ""
    else:
        feedback = response
        if feedback.upper().startswith("REWORK:"):
            feedback = feedback[len("REWORK:") :].strip()
        return False, feedback


with open("/Users/galyukshev/Desktop/NSS/smol-llm-research/data (1).json", "r") as f:
    data = f.readlines()

elems = []
for line in data:
    line = json.loads(line)
    initial_state: OverallState = {
        "user_input": line["user_request"],
        "graph_output": "",
        "reasoning_chain": [],
    }

    selected_category = supervisor_router(initial_state)
    initial_state["graph_output"] += f"\n[Selected Category] {selected_category}"

    iteration = 0
    max_iterations = 5
    approved = False
    feedback = None

    while iteration < max_iterations and not approved:
        initial_state = agent_worker(initial_state, selected_category, feedback)
        approved, feedback = supervisor_evaluate(initial_state)
        if not approved:
            initial_state["graph_output"] += f"\n[Supervisor Feedback] {feedback}"
        iteration += 1

    if approved:
        initial_state[
            "graph_output"
        ] += "\n[Final Approval] The proposed sequence has been approved by the supervisor."
    else:
        initial_state[
            "graph_output"
        ] += (
            "\n[Final Approval] Maximum iterations reached. Final proposal is provided."
        )

    # print("\nFinal Reasoning Chain:")
    # for step in initial_state["reasoning_chain"]:
    #     print(step)
    line["generated"] = final_ans
    elems.append(line)
    with open("res.json", "w", encoding="utf-8") as f:
        json.dump(elems, f, ensure_ascii=False, indent=4)
    # with open("res.txt", "a") as f:
    #     f.write(data["user_request"])
    #     f.write(final_ans, "\n")
    #     print(final_ans)
# print("\nFinal Graph Output:")
# print(initial_state["graph_output"])
