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
    type: str = "object"
    properties: Dict[str, AnyToolProperty] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class ToolSchema(BaseModel):
    name: str
    description: str
    arguments: Optional[ToolIOSchema] = None
    results: Optional[ToolIOSchema] = None


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
    question: str
    reference: List[BenchmarkReference]
    task_nodes: Optional[List[Dict]] = None
    task_links: Optional[List[Dict]] = None
    n_tools: Optional[int] = None
    type: Optional[str] = None


class AgentState(BaseModel):
    user_request: str
    category: Optional[str] = None
    agent_outcome: Optional[List[ToolCall]] = None
    error_message: Optional[str] = None
    supervisor_result: Optional[List[ToolCall]] = None
    all_tools_schema: Optional[Dict[str, ToolSchema]] = None
    tools_by_category: Optional[Dict[str, Dict[str, ToolSchema]]] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    retry_count: int = 0
    available_tools_for_agent: Optional[Dict[str, ToolSchema]] = None

    def clear_error(self):
        self.error_message = None
        return self

    def set_error(self, error: str):
        self.error_message = error
        return self

    def get_current_agent_tools(self) -> Dict[str, ToolSchema]:
        result_dict = {}
        for category in self.tools_by_category.keys():
            result_dict.update(self.tools_by_category.get(category, {}))
        return result_dict

    class Config:
        arbitrary_types_allowed = True


NestedObjectProperty.model_rebuild()
ToolIOSchema.model_rebuild()
