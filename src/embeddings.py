from __future__ import annotations

import hashlib
import math
import os
import dotenv

dotenv.load_dotenv()

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GITHUB_MODELS_TOKEN_ENV = "GITHUB_MODELS_TOKEN"
GITHUB_EMBEDDING_MODEL_ENV = "GITHUB_EMBEDDING_MODEL"
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com/"
OPENAI_EMBEDDING_BATCH_SIZE_ENV = "OPENAI_EMBEDDING_BATCH_SIZE"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self(text) for text in texts]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            return [[float(value) for value in row] for row in embeddings.tolist()]
        return [[float(value) for value in row] for row in embeddings]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(
        self,
        model_name: str = OPENAI_EMBEDDING_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        resolved_api_key = api_key or os.getenv(OPENAI_API_KEY_ENV, "")
        resolved_base_url = base_url or os.getenv(OPENAI_BASE_URL_ENV)
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        configured_batch_size = batch_size or int(os.getenv(OPENAI_EMBEDDING_BATCH_SIZE_ENV, "64"))
        self.batch_size = max(1, configured_batch_size)

    def __call__(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            ordered_items = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            for item in ordered_items:
                all_embeddings.append([float(value) for value in item.embedding])
        return all_embeddings


_mock_embed = MockEmbedder()
