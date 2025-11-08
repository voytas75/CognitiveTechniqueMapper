from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Sequence

try:
    from litellm import embedding as litellm_embedding
except ImportError:  # pragma: no cover - optional dependency
    litellm_embedding = None

from .config_service import ConfigService


class EmbeddingGateway:
    """Generates embeddings with litellm and falls back to deterministic hashes if unavailable."""

    def __init__(self, config_service: ConfigService, *, use_fallback: bool = True) -> None:
        self._config_service = config_service
        self._config = config_service.get_embedding_config()
        self._providers = config_service.providers
        self._use_fallback = use_fallback

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []

        try:
            return self._invoke_litellm(texts_list)
        except Exception:
            if not self._use_fallback:
                raise
            return [self._fallback_embedding(text) for text in texts_list]

    def _invoke_litellm(self, texts: Sequence[str]) -> List[List[float]]:
        if litellm_embedding is None:
            raise RuntimeError("litellm is not installed; cannot request embeddings.")
        params = {"model": self._config.model, "input": list(texts)}
        provider_config = {}
        if self._config.provider:
            provider_config = self._providers.get(self._config.provider, {})
            api_key_env = provider_config.get("api_key_env")
            if api_key_env:
                from os import environ

                api_key = environ.get(api_key_env)
                if not api_key:
                    raise RuntimeError(
                        f"Environment variable '{api_key_env}' required for embeddings provider '{self._config.provider}'."
                    )
                params.setdefault("api_key", api_key)
            for key, value in provider_config.items():
                if key not in {"api_key_env"}:
                    params.setdefault(key, value)

        response = litellm_embedding(**params)
        data = response.get("data")
        if not data:
            raise RuntimeError("Embedding API returned no data")
        return [item.get("embedding", []) for item in data]

    def _fallback_embedding(self, text: str, dimensions: int = 12) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        chunk_size = len(digest) // dimensions
        vector = []
        for idx in range(dimensions):
            start = idx * chunk_size
            end = start + chunk_size
            chunk = digest[start:end]
            value = int.from_bytes(chunk, "big", signed=False)
            scaled = (value % 1000) / 1000.0
            vector.append(scaled)

        norm = math.sqrt(sum(component ** 2 for component in vector)) or 1.0
        return [component / norm for component in vector]
