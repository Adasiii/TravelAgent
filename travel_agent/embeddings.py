from __future__ import annotations

import hashlib
import math

from .config import settings


class HashEmbeddingFunction:
    """Deterministic fallback embeddings for offline development."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def name(self) -> str:
        return "travel-agent-hash-embedding"

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        grams = [text[i : i + 2] for i in range(max(len(text) - 1, 1))]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class DashScopeEmbeddingFunction:
    """Chroma embedding function backed by DashScope text-embedding-v4."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def name(self) -> str:
        return f"dashscope-{self.model}"

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import dashscope
        from http import HTTPStatus

        dashscope.api_key = settings.dashscope_api_key
        resp = dashscope.TextEmbedding.call(model=self.model, input=texts)
        status_code = getattr(resp, "status_code", None)
        if status_code != HTTPStatus.OK:
            message = getattr(resp, "message", None) or str(resp)
            raise RuntimeError(f"DashScope embedding failed: {message}")

        output = getattr(resp, "output", None) or resp.get("output", {})
        embeddings = output.get("embeddings", [])
        return [item["embedding"] for item in embeddings]


def build_embedding_function(prefer_dashscope: bool = True):
    if prefer_dashscope and settings.dashscope_api_key:
        return DashScopeEmbeddingFunction()
    return HashEmbeddingFunction()
