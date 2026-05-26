"""RAG query pipeline: retrieve relevant chunks (hybrid BM25 + Dense + RRF) and ask LLM."""

import json
from pathlib import Path
from typing import List

import jieba
from openai import OpenAI
from rank_bm25 import BM25Okapi

from .store import get_store


def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_llm_client():
    cfg = _load_config()["deepseek"]
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    ), cfg["model"]


# ---------------------------------------------------------------------------
# BM25 index (lazy singleton, rebuilt on first query after reset)
# ---------------------------------------------------------------------------

_bm25_index = None


def _tokenize(text: str) -> List[str]:
    """Tokenize text using jieba (handles both Chinese and English)."""
    return [w for w in jieba.cut(text) if w.strip()]


def _get_bm25_index():
    """Build BM25 index from all documents in the store. Cached after first call."""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    collection, _ = get_store()
    results = collection.get(include=["documents"])

    if not results["ids"]:
        _bm25_index = (None, [], [])
        return _bm25_index

    docs = results["documents"]
    ids = results["ids"]
    metadatas = results.get("metadatas", [{}] * len(ids))

    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    _bm25_index = (bm25, ids, metadatas)
    return _bm25_index


def reset_bm25_index():
    """Force rebuild BM25 index on next query (call after ingestion)."""
    global _bm25_index
    _bm25_index = None


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def _retrieve_dense(question: str, top_k: int) -> List[dict]:
    """Dense retrieval via ChromaDB vector similarity."""
    collection, _ = get_store()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            chunks.append({
                "id": doc_id,
                "content": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
    return chunks


def _retrieve_bm25(question: str, top_k: int) -> List[dict]:
    """BM25 sparse retrieval."""
    bm25, ids, metadatas = _get_bm25_index()
    if bm25 is None:
        return []

    tokens = _tokenize(question)
    scores = bm25.get_scores(tokens)

    # Get top-k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    collection, _ = get_store()
    chunks = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        result = collection.get(ids=[ids[idx]], include=["documents", "metadatas"])
        if result["documents"]:
            chunks.append({
                "id": ids[idx],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "distance": None,
                "bm25_score": float(scores[idx]),
            })
    return chunks


def _rrf_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
    top_k: int = 5,
) -> List[dict]:
    """Reciprocal Rank Fusion: merge multiple ranked result lists.

    RRF_score(d) = sum(1 / (k + rank_i(d))) across all lists.
    """
    scores: dict[str, float] = {}
    id_to_chunk: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            doc_id = chunk["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in id_to_chunk:
                id_to_chunk[doc_id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids[:top_k]:
        chunk = id_to_chunk[doc_id]
        chunk["rrf_score"] = scores[doc_id]
        results.append(chunk)
    return results


def _expand_to_parent(chunk: dict) -> dict:
    """If chunk has parent_content metadata, expand content to parent text."""
    parent = chunk.get("metadata", {}).get("parent_content")
    if parent:
        return {**chunk, "content": parent}
    return chunk


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------


def retrieve(question: str, top_k: int | None = None) -> List[dict]:
    """Search the knowledge base and return top-k relevant chunks.

    Supports hybrid retrieval (BM25 + Dense + RRF fusion) when enabled
    in config. Automatically expands child chunks to parent context
    when parent-child chunking is used.
    """
    cfg = _load_config()["rag"]
    if top_k is None:
        top_k = cfg["top_k"]

    hybrid = cfg.get("hybrid_search", False)

    if hybrid:
        # Retrieve more candidates from each source, then fuse
        fetch_k = top_k * 4
        dense_results = _retrieve_dense(question, fetch_k)
        bm25_results = _retrieve_bm25(question, fetch_k)
        chunks = _rrf_fusion([dense_results, bm25_results], top_k=top_k)
    else:
        chunks = _retrieve_dense(question, top_k)

    # Expand child chunks to parent content if parent-child chunking was used
    chunks = [_expand_to_parent(c) for c in chunks]

    return chunks


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


RAG_PROMPT = r"""你是一个知识库问答助手。请根据以下检索到的知识库内容回答用户问题。

规则：
1. 如果知识库内容足以回答问题，请基于内容给出准确、完整的回答。
2. 如果知识库内容只提供部分信息，请基于已有信息回答，并说明哪些部分无法确定。
3. 如果知识库内容与问题完全无关，请如实告知用户"知识库中没有找到相关信息"。
4. 回答时尽量引用知识库中的具体内容。
5. 使用与用户提问相同的语言回答。

**数学公式格式要求（重要）：**
- 所有数学公式、变量、希腊字母（如 Theta、sum、Omega 等）必须用 $...$（行内）或 $$...$$（独立行）包裹。
- 正确示例：时间复杂度为 $\Theta(n)$，公式 $$T(n) = c_1 + c_2 \cdot n$$
- 错误示例：时间复杂度为 \Theta(n)，公式 T(n) = c_1 + c_2 * n
- 下标用 _（如 $c_1$），乘号用 \cdot（如 $c_2 \cdot n$）
- 即使只是单个希腊字母也必须加 $，如 $\Theta$、$\Omega$

## 知识库内容

{context}

## 用户问题

{question}

## 回答
"""


def ask(question: str, top_k: int | None = None) -> str:
    """Full RAG pipeline: retrieve + LLM answer."""
    chunks = retrieve(question, top_k)

    if not chunks:
        return "知识库中暂无内容，请先导入一些文档。"

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        src = chunk["metadata"].get("source_name", "unknown")
        context_parts.append(f"[文档 {i+1}: {src}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    client, model = _get_llm_client()
    prompt = RAG_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是知识库问答助手，基于提供的资料回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content or ""

    # Append sources
    sources = set(
        chunk["metadata"].get("source_name", "?")
        for chunk in chunks
    )
    if sources:
        answer += f"\n\n---\n参考来源: {', '.join(sorted(sources))}"

    return answer
