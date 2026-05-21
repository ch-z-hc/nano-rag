"""RAG query pipeline: retrieve relevant chunks and ask LLM."""

import json
from pathlib import Path
from typing import List

from openai import OpenAI

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


def retrieve(question: str, top_k: int | None = None) -> List[dict]:
    """Search the knowledge base and return top-k relevant chunks with metadata."""
    cfg = _load_config()["rag"]
    if top_k is None:
        top_k = cfg["top_k"]

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
