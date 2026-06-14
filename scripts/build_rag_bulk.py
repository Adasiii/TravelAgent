"""批量构建 RAG：通过 MediaWiki 官方 API 拉取 Wikivoyage 目的地词条正文，
切块后写入 Chroma（DashScope text-embedding-v4）。

仅使用 Wikivoyage（CC BY-SA 授权，允许复用）的公开 API，不抓取任何
需要登录或禁止抓取的站点（小红书 / 知乎等）。

用法：
    conda run -n travelagent python scripts/build_rag_bulk.py --target 1000
    conda run -n travelagent python scripts/build_rag_bulk.py --lang en --target 1000 --reset
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from travel_agent.config import settings
from travel_agent.embeddings import DashScopeEmbeddingFunction

USER_AGENT = "TravelAgentRAG/0.1 (educational RAG prototype; contact: local)"

API_BASE = {
    "zh": "https://zh.wikivoyage.org/w/api.php",
    "en": "https://en.wikivoyage.org/w/api.php",
}
PAGE_URL = {
    "zh": "https://zh.wikivoyage.org/wiki/",
    "en": "https://en.wikivoyage.org/wiki/",
}


def get_json(session: requests.Session, url: str, params: dict, retries: int = 4) -> dict:
    """带 429/网络错误指数退避的 GET。"""
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                print(f"[rate] 429 限速，等待 {wait:.1f}s...", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"[retry] 请求失败({exc})，{delay:.1f}s 后重试...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return {}


def list_titles(lang: str, max_titles: int, session: requests.Session) -> list[str]:
    """阶段一：用 list=allpages 快速枚举主命名空间词条标题（每次 500 个）。"""
    api = API_BASE[lang]
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": "500",
        "apfilterredir": "nonredirects",
    }
    titles: list[str] = []
    while len(titles) < max_titles:
        data = get_json(session, api, params)
        titles.extend(p["title"] for p in (data.get("query") or {}).get("allpages", []))
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(0.3)
    return titles[:max_titles]


def fetch_extract(lang: str, title: str, session: requests.Session) -> str:
    """阶段二：单标题拉取完整纯文本正文（单页一次请求即可拿全文）。"""
    data = get_json(
        session,
        API_BASE[lang],
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "titles": title,
        },
    )
    pages = (data.get("query") or {}).get("pages") or {}
    if not pages:
        return ""
    return (next(iter(pages.values())).get("extract") or "").strip()


def fetch_pages(lang: str, max_pages: int, min_chars: int, session: requests.Session):
    """枚举标题后逐个取正文，产出 (title, url, text)，由调用方决定何时停止。"""
    titles = list_titles(lang, max_pages, session)
    print(f"[fetch] 枚举到 {len(titles)} 个词条标题，开始逐个取正文...", flush=True)
    kept = 0
    for i, title in enumerate(titles, start=1):
        try:
            text = fetch_extract(lang, title, session)
        except Exception as exc:
            print(f"[warn] 取正文失败 {title}: {exc}", flush=True)
            continue
        if len(text) < min_chars:
            continue
        kept += 1
        yield title, PAGE_URL[lang] + title.replace(" ", "_"), text
        time.sleep(0.4)  # 对 Wikimedia 友好限速
    print(f"[fetch] 共保留 {kept} 个有足够正文的词条。", flush=True)


def stable_id(text: str, source: str, idx: int) -> str:
    digest = hashlib.sha1(f"{source}:{idx}:{text[:80]}".encode("utf-8")).hexdigest()
    return f"{source}:{idx}:{digest}"


def build(lang: str, target_chunks: int, min_chars: int, max_pages: int, reset: bool) -> int:
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for text-embedding-v4 RAG indexing.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    client = chromadb.PersistentClient(path=settings.chroma_path)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            print(f"[reset] 已删除旧 collection {settings.chroma_collection}", flush=True)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=DashScopeEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    raw_dir = Path("data/downloads_bulk")
    raw_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metas: list[dict] = []

    def flush() -> int:
        if not batch_texts:
            return 0
        try:
            collection.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)
            n = len(batch_texts)
        except Exception as exc:
            print(f"[warn] 写入一批失败，跳过：{exc}", flush=True)
            n = 0
        batch_ids.clear()
        batch_texts.clear()
        batch_metas.clear()
        return n

    for title, url, text in fetch_pages(lang, max_pages, min_chars, session):
        # 保存原始正文，便于审查来源
        safe = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        (raw_dir / f"{safe}.txt").write_text(f"Source: {url}\n\n{text}", encoding="utf-8")

        for idx, chunk in enumerate(splitter.split_text(text)):
            chunk = chunk.strip()
            if not chunk:
                continue
            batch_ids.append(stable_id(chunk, url, idx))
            batch_texts.append(chunk)
            batch_metas.append({"source": title, "url": url, "lang": lang, "chunk": idx})
            if len(batch_texts) >= 10:           # embedding 批量大小
                total += flush()
                if total and total % 100 == 0:
                    print(f"[index] 已写入约 {total} 块...", flush=True)
        if total >= target_chunks:
            break

    total += flush()
    print(f"[done] 共写入 {total} 个文本块 -> {settings.chroma_path}/{settings.chroma_collection}", flush=True)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-build Chroma RAG from Wikivoyage MediaWiki API.")
    p.add_argument("--lang", choices=["zh", "en"], default="zh", help="Wikivoyage 语言版本")
    p.add_argument("--target", type=int, default=1000, help="目标文本块数量（达到即停止）")
    p.add_argument("--min-chars", type=int, default=400, help="词条正文最小字符数（过滤小作品/消歧页）")
    p.add_argument("--max-pages", type=int, default=2000, help="最多扫描多少个词条（防止无限拉取）")
    p.add_argument("--reset", action="store_true", help="先删除旧 collection 再重建")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.lang, args.target, args.min_chars, args.max_pages, args.reset)
