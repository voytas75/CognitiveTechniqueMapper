"""Embedding gateway for technique vectors.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and expanded method documentation.
"""

from __future__ import annotations

import hashlib
import math
import logging
from typing import Iterable, List, Sequence

try:
    import litellm
    from litellm import embedding as litellm_embedding
except ImportError:  # pragma: no cover - optional dependency
    litellm_embedding = None
else:  # pragma: no cover - attribute may not exist on older versions
    try:
        litellm.drop_params = True
    except AttributeError:
        pass

from .config_service import ConfigService

logger = logging.getLogger(__name__)


class EmbeddingGateway:
    """Generates embeddings with litellm and falls back to deterministic hashes if unavailable."""

    def __init__(
        self, config_service: ConfigService, *, use_fallback: bool = True
    ) -> None:
        """Initialize embedding resources.

        Args:
            config_service (ConfigService): Configuration provider for embedding models.
            use_fallback (bool): Enable hashed fallback vectors when providers fail.
        """

        self._config_service = config_service
        self._config = config_service.get_embedding_config()
        self._providers = config_service.providers
        self._use_fallback = use_fallback

    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for a single text block.

        Args:
            text (str): Input text for embedding.

        Returns:
            list[float]: Embedding vector returned by the provider or fallback.
        """

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts.

        Args:
            texts (Iterable[str]): Sequence of text snippets to embed.

        Returns:
            list[list[float]]: Embedding vectors aligned with the input order.

        Raises:
            RuntimeError: When embedding fails and fallback is disabled.
        """

        texts_list = list(texts)
        if not texts_list:
            return []

        try:
            return self._invoke_litellm(texts_list)
        except Exception as exc:
            logger.warning("Embedding generation failed (%s); using fallback.", exc)
            if not self._use_fallback:
                raise RuntimeError(f"Embedding generation failed: {exc}") from exc
            return [self._fallback_embedding(text) for text in texts_list]

    def _invoke_litellm(self, texts: Sequence[str]) -> List[List[float]]:
        """Call litellm to create embeddings for the supplied texts.

        Args:
            texts (Sequence[str]): Texts to embed using the configured provider.

        Returns:
            list[list[float]]: Embedding vectors from the provider.

        Raises:
            RuntimeError: If the provider configuration is invalid or returns no data.
        """

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
                    message = f"Environment variable '{api_key_env}' required for embeddings provider '{self._config.provider}'."
                    logger.error(message)
                    raise RuntimeError(message)
                params.setdefault("api_key", api_key)
            for key, value in provider_config.items():
                if key not in {"api_key_env"}:
                    params.setdefault(key, value)

        response = litellm_embedding(**params)
        data = response.get("data")
        if not data:
            raise RuntimeError("Embedding API returned no data")
        logger.debug(
            "Received %s embedding vectors from %s",
            len(data),
            self._config.model,
        )
        return [item.get("embedding", []) for item in data]

    def _fallback_embedding(self, text: str, dimensions: int = 12) -> List[float]:
        """Generate a deterministic fallback embedding vector.

        Args:
            text (str): Source text to hash.
            dimensions (int): Number of components to produce.

        Returns:
            list[float]: Normalized pseudo-embedding vector.
        """

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

        norm = math.sqrt(sum(component**2 for component in vector)) or 1.0
        return [component / norm for component in vector]
