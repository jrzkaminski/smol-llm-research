from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field


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
    description_expanded: Optional[str] = None
    synthetic_questions: Optional[List[str]] = None


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
