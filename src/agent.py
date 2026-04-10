from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self._store.search(question, top_k=top_k)
        context_blocks = []
        for i, chunk in enumerate(retrieved, start=1):
            score = chunk.get("score", 0.0)
            content = chunk.get("content", "")
            context_blocks.append(f"[Chunk {i} | score={score:.4f}]\n{content}")

        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."
        prompt = (
            "You are a helpful QA assistant. Answer the question using only the provided context.\n"
            "If the context is insufficient, say that you do not have enough information.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context_text}\n\n"
            "Answer:"
        )
        return self._llm_fn(prompt)
