from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("USER_AGENT", "TravelAgentRAG/0.1")

import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from travel_agent.config import settings
from travel_agent.embeddings import DashScopeEmbeddingFunction


DEFAULT_URLS = [
    "https://en.wikivoyage.org/wiki/Hangzhou",
    "https://en.wikivoyage.org/wiki/Shanghai",
    "https://en.wikivoyage.org/wiki/Beijing",
    "https://en.wikivoyage.org/wiki/Suzhou",
]


def stable_id(text: str, source: str, idx: int) -> str:
    digest = hashlib.sha1(f"{source}:{idx}:{text[:80]}".encode("utf-8")).hexdigest()
    return f"{source}:{idx}:{digest}"


def load_web_documents(urls: list[str]):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = url
        docs.extend(loaded)
    return docs


def save_downloads(docs, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", f"doc-{idx}")
        name = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        target = output_dir / f"{idx:02d}_{name}.txt"
        target.write_text(f"Source: {source}\n\n{doc.page_content}", encoding="utf-8")


def build_index(urls: list[str], reset: bool = False) -> int:
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for text-embedding-v4 RAG indexing.")

    docs = load_web_documents(urls)
    save_downloads(docs, Path("data/downloads"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    client = chromadb.PersistentClient(path=settings.chroma_path)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=DashScopeEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    texts = []
    metas = []
    for idx, doc in enumerate(chunks):
        source = doc.metadata.get("source", "")
        text = doc.page_content.strip()
        if not text:
            continue
        ids.append(stable_id(text, source, idx))
        texts.append(text)
        metas.append({"source": source, "url": source, "chunk": idx})

    batch_size = 10
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metas[start:end],
        )
    return len(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TravelAgent Chroma HNSW RAG index from web pages.")
    parser.add_argument("--url", action="append", dest="urls", help="URL to load. Can be used multiple times.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    urls = args.urls or DEFAULT_URLS
    count = build_index(urls, reset=args.reset)
    print(f"Indexed {count} chunks into {settings.chroma_path}/{settings.chroma_collection}")
