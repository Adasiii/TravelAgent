from __future__ import annotations

from pathlib import Path

from .config import settings
from .embeddings import build_embedding_function


class LocalRAG:
    def __init__(
        self,
        persist_path: str | None = None,
        collection_name: str | None = None,
        prefer_dashscope_embedding: bool = True,
    ):
        self.persist_path = persist_path or settings.chroma_path
        self.collection_name = collection_name or settings.chroma_collection
        self.available = False
        self.error = ""
        self._collection = None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_path)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=build_embedding_function(prefer_dashscope_embedding),
                metadata={"hnsw:space": "cosine"},
            )
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def ingest_folder(self, folder: str | None = None) -> int:
        folder_path = Path(folder or settings.knowledge_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        if not self.available or self._collection is None:
            return 0

        count = 0
        for file_path in folder_path.glob("**/*"):
            if file_path.suffix.lower() not in {".md", ".txt"} or not file_path.is_file():
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk_text(text)
            if not chunks:
                continue
            ids = [f"{file_path.as_posix()}::{i}" for i in range(len(chunks))]
            metas = [{"source": file_path.name, "chunk": i, "url": ""} for i in range(len(chunks))]
            self._collection.upsert(ids=ids, documents=chunks, metadatas=metas)
            count += len(chunks)
        return count

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.available or self._collection is None:
            return []
        try:
            result = self._collection.query(query_texts=[query], n_results=k)
        except Exception as exc:
            self.error = str(exc)
            return []

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {"text": doc, "metadata": meta or {}, "distance": distance}
            for doc, meta, distance in zip(docs, metas, distances)
        ]

    @staticmethod
    def _chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not cleaned:
            return []
        chunks = []
        start = 0
        while start < len(cleaned):
            chunks.append(cleaned[start : start + size])
            start += max(size - overlap, 1)
        return chunks
