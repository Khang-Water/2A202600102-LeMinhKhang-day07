from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": f"{doc.id}_{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_embed = getattr(self._embedding_fn, "embed_texts", None)
        if callable(batch_embed):
            embedded = batch_embed(texts)
            if len(embedded) != len(texts):
                raise ValueError("embed_texts must return exactly one embedding per input text")
            return embedded
        return [self._embedding_fn(text) for text in texts]

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata.setdefault("doc_id", doc.id)
            ids.append(f"{doc.id}_{self._next_index}")
            documents.append(doc.content)
            metadatas.append(metadata)
            self._next_index += 1

        embeddings = self._embed_many(documents)

        if self._use_chroma and self._collection is not None:
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            return

        for idx in range(len(ids)):
            self._store.append(
                {
                    "id": ids[idx],
                    "content": documents[idx],
                    "metadata": metadatas[idx],
                    "embedding": embeddings[idx],
                }
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            if self.get_collection_size() == 0:
                return []

            query_embedding = self._embedding_fn(query)
            raw = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            ids = (raw.get("ids") or [[]])[0]
            docs = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            results: list[dict[str, Any]] = []
            for i in range(len(docs)):
                score = -distances[i] if i < len(distances) else 0.0
                results.append(
                    {
                        "id": ids[i] if i < len(ids) else "",
                        "content": docs[i],
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "score": score,
                    }
                )
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]

        return self._search_records(query=query, records=self._store, top_k=top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if top_k <= 0:
            return []
        if not metadata_filter:
            return self.search(query=query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            raw = self._collection.query(query_embeddings=[query_embedding], n_results=top_k, where=metadata_filter)
            ids = (raw.get("ids") or [[]])[0]
            docs = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            results: list[dict[str, Any]] = []
            for i in range(len(docs)):
                score = -distances[i] if i < len(distances) else 0.0
                results.append(
                    {
                        "id": ids[i] if i < len(ids) else "",
                        "content": docs[i],
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "score": score,
                    }
                )
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]

        filtered_records = [
            record
            for record in self._store
            if all(record.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query=query, records=filtered_records, top_k=top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            matches = self._collection.get(where={"doc_id": doc_id}, include=[])
            ids = matches.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                return True
            return False

        before = len(self._store)
        self._store = [record for record in self._store if record.get("metadata", {}).get("doc_id") != doc_id]
        return len(self._store) < before
