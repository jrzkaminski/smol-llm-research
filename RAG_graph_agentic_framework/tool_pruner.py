from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from RAG_graph_agentic_framework.config import OPENAI_API_KEY
from RAG_graph_agentic_framework.schemas import ToolSchema


VECTORSTORE_DIR = Path("/tmp/chroma_global_tools")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

_GLOBAL_VS: Chroma | None = None
_GLOBAL_TOOL_CACHE: Dict[str, ToolSchema] = {}


def _build_vectorstore(tools: Dict[str, ToolSchema]) -> Chroma:
    docs: List[Document] = []

    for tool_name, schema in tools.items():
        arg_parts: List[str] = []
        if schema.arguments and schema.arguments.properties:
            for arg_name, arg_prop in schema.arguments.properties.items():
                arg_desc = arg_prop.description or ""
                arg_parts.append(f"{arg_name}: {arg_prop.type} — {arg_desc}")
        arg_block = " | ".join(arg_parts) if arg_parts else "none"

        page_text = f"{tool_name}\n{schema.description}\nArguments: {arg_block}"
        docs.append(Document(page_content=page_text, metadata={"tool_name": tool_name}))

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="all_tools",
        persist_directory=str(VECTORSTORE_DIR),
    )


def init_pruner(tools: Dict[str, ToolSchema]) -> None:
    global _GLOBAL_VS, _GLOBAL_TOOL_CACHE
    if _GLOBAL_VS is None:
        _GLOBAL_TOOL_CACHE = tools
        _GLOBAL_VS = _build_vectorstore(tools)
        print(f"[Pruner] Was init for {len(tools)} tools")


def get_top_tool_schemas(user_query: str, top_k: int = 40) -> Dict[str, ToolSchema]:
    if _GLOBAL_VS is None:
        raise RuntimeError("Pruner wasn't init.")

    hits = _GLOBAL_VS.similarity_search(user_query, k=top_k)
    unique_names: List[str] = []
    seen = set()
    for doc in hits:
        name = doc.metadata["tool_name"]
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    return {name: _GLOBAL_TOOL_CACHE[name] for name in unique_names}
