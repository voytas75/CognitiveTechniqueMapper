"""Explicit service factory functions for the CLI runtime."""

from __future__ import annotations

from typing import Any, Optional

from src.cli.io import console
from src.cli.state import PROJECT_ROOT
from src.db.sqlite_client import SQLiteClient
from src.services.config_service import ConfigService
from src.services.data_initializer import TechniqueDataInitializer
from src.services.embedding_gateway import EmbeddingGateway
from src.services.technique_catalog import TechniqueCatalogService
from src.services.technique_search import TechniqueSearchService

try:
    from src.db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore[assignment]


def create_optional_chroma_client(
    persist_directory: str,
    collection_name: str,
    *,
    client_cls: Any = ChromaClient,
) -> Optional[Any]:
    """Create an optional Chroma client and degrade gracefully on failure.

    Args:
        persist_directory: Filesystem path used for Chroma persistence.
        collection_name: Chroma collection name.
        client_cls: Chroma client class, or ``None`` to disable Chroma.

    Returns:
        Initialized Chroma client, or ``None`` when disabled or unavailable.
    """

    if client_cls is None:
        return None
    try:
        return client_cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        console.print(f"[yellow]ChromaDB disabled: {exc}[/]")
        return None


def create_catalog_service(
    *,
    config_service_cls: Any = ConfigService,
    sqlite_client_cls: Any = SQLiteClient,
    embedding_gateway_cls: Any = EmbeddingGateway,
    technique_catalog_service_cls: Any = TechniqueCatalogService,
    chroma_client_cls: Any = ChromaClient,
) -> tuple[TechniqueCatalogService, SQLiteClient]:
    """Instantiate a catalog service with explicit, overrideable dependencies.

    Args:
        config_service_cls: Factory for application configuration.
        sqlite_client_cls: Factory for the SQLite catalog client.
        embedding_gateway_cls: Factory for the embedding gateway.
        technique_catalog_service_cls: Factory for the catalog service.
        chroma_client_cls: Chroma client class, or ``None`` to disable Chroma.

    Returns:
        Configured catalog service and its SQLite client.
    """

    config_service = config_service_cls()
    db_config = config_service.database_config
    sqlite_client = sqlite_client_cls(
        db_config.get("sqlite_path", "./data/techniques.db")
    )
    sqlite_client.initialize_schema()
    chroma_client = create_optional_chroma_client(
        db_config.get("chromadb_path", "./embeddings"),
        db_config.get("chromadb_collection", "techniques"),
        client_cls=chroma_client_cls,
    )
    embedder = embedding_gateway_cls(config_service=config_service)
    dataset_path = PROJECT_ROOT / "data" / "techniques.json"
    catalog = technique_catalog_service_cls(
        sqlite_client=sqlite_client,
        embedder=embedder,
        dataset_path=dataset_path,
        chroma_client=chroma_client,
    )
    return catalog, sqlite_client


def create_initializer(
    *,
    config_service_cls: Any = ConfigService,
    sqlite_client_cls: Any = SQLiteClient,
    embedding_gateway_cls: Any = EmbeddingGateway,
    initializer_cls: Any = TechniqueDataInitializer,
    chroma_client_cls: Any = ChromaClient,
) -> tuple[TechniqueDataInitializer, SQLiteClient]:
    """Create a dataset initializer with explicit, overrideable dependencies.

    Args:
        config_service_cls: Factory for application configuration.
        sqlite_client_cls: Factory for the SQLite catalog client.
        embedding_gateway_cls: Factory for the embedding gateway.
        initializer_cls: Factory for the dataset initializer.
        chroma_client_cls: Chroma client class, or ``None`` to disable Chroma.

    Returns:
        Configured dataset initializer and its SQLite client.
    """

    config_service = config_service_cls()
    db_config = config_service.database_config
    sqlite_client = sqlite_client_cls(
        db_config.get("sqlite_path", "./data/techniques.db")
    )
    sqlite_client.initialize_schema()
    chroma_client = create_optional_chroma_client(
        db_config.get("chromadb_path", "./embeddings"),
        db_config.get("chromadb_collection", "techniques"),
        client_cls=chroma_client_cls,
    )
    embedder = embedding_gateway_cls(config_service=config_service)
    dataset_path = PROJECT_ROOT / "data" / "techniques.json"
    initializer = initializer_cls(
        sqlite_client=sqlite_client,
        embedder=embedder,
        chroma_client=chroma_client,
        dataset_path=dataset_path,
    )
    return initializer, sqlite_client


def create_search_service(
    *,
    config_service_cls: Any = ConfigService,
    sqlite_client_cls: Any = SQLiteClient,
    embedding_gateway_cls: Any = EmbeddingGateway,
    technique_search_service_cls: Any = TechniqueSearchService,
    chroma_client_cls: Any = ChromaClient,
) -> tuple[TechniqueSearchService, SQLiteClient]:
    """Instantiate a search service with explicit, overrideable dependencies.

    Args:
        config_service_cls: Factory for application configuration.
        sqlite_client_cls: Factory for the SQLite catalog client.
        embedding_gateway_cls: Factory for the embedding gateway.
        technique_search_service_cls: Factory for the search service.
        chroma_client_cls: Chroma client class, or ``None`` to disable Chroma.

    Returns:
        Configured search service and its SQLite client.
    """

    config_service = config_service_cls()
    db_config = config_service.database_config
    sqlite_client = sqlite_client_cls(
        db_config.get("sqlite_path", "./data/techniques.db")
    )
    sqlite_client.initialize_schema()
    chroma_client = create_optional_chroma_client(
        db_config.get("chromadb_path", "./embeddings"),
        db_config.get("chromadb_collection", "techniques"),
        client_cls=chroma_client_cls,
    )
    embedder = embedding_gateway_cls(config_service=config_service)
    service = technique_search_service_cls(
        sqlite_client=sqlite_client,
        embedder=embedder,
        chroma_client=chroma_client,
    )
    return service, sqlite_client


__all__ = [
    "create_catalog_service",
    "create_initializer",
    "create_optional_chroma_client",
    "create_search_service",
]
