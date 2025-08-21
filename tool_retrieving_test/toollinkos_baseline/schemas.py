from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field


class GraphLink(BaseModel):
    source: str
    target: str


class GraphNode(BaseModel):
    name: str
    description: str
    category: str


class GraphStructure(BaseModel):
    links: List[GraphLink]
    nodes: List[GraphNode]


AnyToolProperty = Union["SimpleToolProperty", "NestedObjectProperty"]


class BaseToolProperty(BaseModel):
    description: Optional[str] = None


class SimpleToolProperty(BaseToolProperty):
    type: str


class NestedObjectProperty(BaseToolProperty):
    type: str = "object"
    properties: Dict[str, AnyToolProperty] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class ToolIOSchema(BaseModel):
    name: str
    type: str = "object"
    description: str
    required: Optional[bool] = False


class DependenceSchema(BaseModel):
    name: str
    dependence_type: str
    reason: str


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ToolIOSchema] = None


class ToolCall(BaseModel):
    tool: str
    param: Dict[str, Any] = Field(default_factory=dict)
    input_source: Optional[str] = None


class BenchmarkReference(BaseModel):
    tool: str
    param: Dict[str, Any] = Field(default_factory=dict)
    input_source: Optional[str] = None


class BenchmarkItem(BaseModel):
    id: Optional[str] = None
    user_query: str
    main_golden_function_name: str
    golden_function_names: List[str]


class AgentState(BaseModel):
    user_request: str
    agent_outcome: Optional[List[ToolCall]] = None
    error_message: Optional[str] = None
    subtasks: List[str] = []
    all_tools_schema: Optional[Dict[str, ToolSchema]] = None
    tools_by_category: Optional[Dict[str, Dict[str, ToolSchema]]] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    retry_count: int = 0
    tools_graph: GraphStructure = None
    available_tools_for_agent: Optional[Dict[str, ToolSchema]] = None
    reference_tool_names: List[str] = []
    noise_level: int = 0
    retrieval_recall: Optional[float] = None

    def clear_error(self):
        self.error_message = None
        return self

    def set_error(self, error: str):
        self.error_message = error
        return self

    def get_agent_tools(self) -> Dict[str, ToolSchema]:
        if self.available_tools_for_agent is not None:
            return self.available_tools_for_agent

        union: Dict[str, ToolSchema] = {}
        if self.tools_by_category:
            for cat_dict in self.tools_by_category.values():
                union.update(cat_dict)
        return union

    class Config:
        arbitrary_types_allowed = True


NestedObjectProperty.model_rebuild()
ToolIOSchema.model_rebuild()
