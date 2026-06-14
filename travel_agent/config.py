from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_model: str = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-pro")
    mcp_mode: str = os.getenv("MCP_MODE", "mock")
    mcp_web_search_endpoint: str = os.getenv(
        "MCP_WEB_SEARCH_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
    )
    mcp_amap_endpoint: str = os.getenv("MCP_AMAP_ENDPOINT", "")
    mcp_auth_header: str = os.getenv("MCP_AUTH_HEADER", "Authorization")
    mcp_auth_scheme: str = os.getenv("MCP_AUTH_SCHEME", "Bearer")
    amap_api_key: str = os.getenv("AMAP_API_KEY", "")
    chroma_path: str = os.getenv("CHROMA_PATH", "data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "travel_knowledge_dashscope_v4")
    knowledge_path: str = os.getenv("KNOWLEDGE_PATH", "data/knowledge")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")


settings = Settings()
