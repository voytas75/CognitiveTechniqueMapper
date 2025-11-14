"""Interactive technique catalog management service.

Updates:
    v0.3.0 - 2025-05-09 - Introduced service for CRUD actions and embedding sync.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..db.sqlite_client import SQLiteClient
from .data_initializer import DEFAULT_DATASET_PATH, TechniqueDataInitializer
from .embedding_gateway import EmbeddingGateway
from .technique_utils import compose_embedding_text

try:
    from ..db.chroma_client import ChromaClient, EmbeddingRecord
except RuntimeError:  # pragma: no cover - optional dependency not available
    ChromaClient = None  # type: ignore
    EmbeddingRecord = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TechniqueCatalogService:
    """Coordinates CRUD operations across SQLite, dataset files, and embeddings."""

    sqlite_client: SQLiteClient
    embedder: EmbeddingGateway
    dataset_path: Path = DEFAULT_DATASET_PATH
    chroma_client: Optional[ChromaClient] = None

    def list(self) -> list[dict[str, Any]]:
        """Return all techniques sorted by name."""

        rows = self.sqlite_client.fetch_all()
        entries = [dict(row) for row in rows]
        entries.sort(key=lambda item: (item.get("name") or "").lower())
        return entries

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add a new technique and synchronize backing stores."""

        name = (payload.get("name") or "").strip()
        description = (payload.get("description") or "").strip()
        if not name or not description:
            raise ValueError("Name and description are required to add a technique.")

        if self.sqlite_client.fetch_by_name(name):
            raise ValueError(f"Technique '{name}' already exists.")

        origin_year = self._coerce_int(payload.get("origin_year"))
        creator = self._normalize_optional(payload.get("creator"))
        category = self._normalize_optional(payload.get("category"))
        core_principles = self._normalize_optional(payload.get("core_principles"))

        self.sqlite_client.insert_technique(
            name=name,
            description=description,
            origin_year=origin_year,
            creator=creator,
            category=category,
            core_principles=core_principles,
        )

        entry = {
            "name": name,
            "description": description,
            "origin_year": origin_year,
            "creator": creator,
            "category": category,
            "core_principles": core_principles,
        }
        dataset = self._load_dataset()
        dataset.append(entry)
        self._write_dataset(dataset)
        self._sync_embedding(entry)
        logger.info(
            "technique_added",
            extra={"tool": "technique_catalog", "technique": name},
        )
        return entry

    def update(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update an existing technique and propagate changes."""

        existing_row = self.sqlite_client.fetch_by_name(name)
        if not existing_row:
            raise ValueError(f"Technique '{name}' not found.")

        normalized_updates = self._prepare_update_payload(updates)
        if not normalized_updates:
            return dict(existing_row)

        new_name = normalized_updates.get("name") or existing_row["name"]
        if new_name and new_name.lower() != existing_row["name"].lower():
            conflict = self.sqlite_client.fetch_by_name(new_name)
            if conflict:
                raise ValueError(f"Technique '{new_name}' already exists.")

        self.sqlite_client.update_technique(name, normalized_updates)

        dataset = self._load_dataset()
        updated_entry = None
        for index, entry in enumerate(dataset):
            if entry.get("name", "").lower() == name.lower():
                updated = {**entry, **normalized_updates}
                dataset[index] = updated
                updated_entry = updated
                break

        if updated_entry is None:
            dataset.append({**dict(existing_row), **normalized_updates})
            updated_entry = dataset[-1]

        self._write_dataset(dataset)
        self._sync_embedding(updated_entry, previous_name=name)
        logger.info(
            "technique_updated",
            extra={
                "tool": "technique_catalog",
                "technique": name,
                "updated_technique": new_name,
            },
        )
        return updated_entry

    def remove(self, name: str) -> None:
        """Remove a technique from all backing stores."""

        if not self.sqlite_client.delete_technique(name):
            raise ValueError(f"Technique '{name}' not found.")

        dataset = self._load_dataset()
        dataset = [entry for entry in dataset if entry.get("name", "").lower() != name.lower()]
        self._write_dataset(dataset)
        self._delete_embedding(name)
        logger.info(
            "technique_removed",
            extra={"tool": "technique_catalog", "technique": name},
        )

    def export_to_file(self, destination: Path | str) -> tuple[Path, int]:
        """Write the current dataset to disk and return the export metadata."""

        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load_dataset()
        dest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "technique_export_completed",
            extra={"tool": "technique_catalog", "technique_count": len(records)},
        )
        return dest_path, len(records)

    def import_from_file(
        self,
        source: Path | str,
        *,
        mode: str = "replace",
        rebuild_embeddings: bool = True,
    ) -> dict[str, int]:
        """Import techniques from a file, optionally appending to the catalog."""

        valid_modes = {"replace", "append"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported import mode '{mode}'. Choose from {sorted(valid_modes)}.")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Import file must contain a JSON list of technique objects.")

        normalized_records = [self._normalize_record(item) for item in payload]

        existing_dataset = self._load_dataset()
        stats = {"added": 0, "updated": 0}

        if mode == "replace":
            combined_dataset = normalized_records
            stats["added"] = len(normalized_records)
            stats["updated"] = 0
        else:
            lookup = {entry.get("name", "").lower(): dict(entry) for entry in existing_dataset}
            order: list[str] = [entry.get("name", "").lower() for entry in existing_dataset]
            for record in normalized_records:
                key = record["name"].lower()
                if key in lookup:
                    lookup[key] = record
                    stats["updated"] += 1
                else:
                    lookup[key] = record
                    order.append(key)
                    stats["added"] += 1
            combined_dataset = [lookup[key] for key in order]

        self._write_dataset(combined_dataset)

        initializer = TechniqueDataInitializer(
            sqlite_client=self.sqlite_client,
            embedder=self.embedder,
            chroma_client=self.chroma_client,
            dataset_path=self.dataset_path,
        )
        initializer.refresh(rebuild_embeddings=rebuild_embeddings)

        logger.info(
            "technique_import_completed",
            extra={
                "tool": "technique_catalog",
                "import_mode": mode,
                "added": stats["added"],
                "updated": stats["updated"],
                "total": len(combined_dataset),
            },
        )

        return {"added": stats["added"], "updated": stats["updated"], "total": len(combined_dataset)}

    def _prepare_update_payload(self, updates: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}

        for key in ("name", "description", "creator", "category", "core_principles"):
            if key in updates and updates[key] is not None:
                prepared[key] = updates[key].strip()

        if "origin_year" in updates:
            prepared["origin_year"] = self._coerce_int(updates.get("origin_year"))

        return {k: v for k, v in prepared.items() if v not in {None, ""}}

    def _sync_embedding(self, entry: dict[str, Any], *, previous_name: str | None = None) -> None:
        if not self.chroma_client or EmbeddingRecord is None:
            return

        text = compose_embedding_text(entry)
        embedding_vector = self.embedder.embed(text)
        record = EmbeddingRecord(
            identifier=entry.get("name", ""),
            embedding=embedding_vector,
            metadata={
                "name": entry.get("name", ""),
                "category": entry.get("category", ""),
                "creator": entry.get("creator", ""),
                "origin_year": str(entry.get("origin_year", "")),
            },
            document=entry.get("description", ""),
        )

        if previous_name and previous_name != record.identifier:
            try:
                self.chroma_client.delete([previous_name])
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "embedding_delete_failed",
                    extra={
                        "tool": "technique_catalog",
                        "technique": previous_name,
                        "error": str(exc),
                    },
                )

        self.chroma_client.upsert_embeddings([record])

    def _delete_embedding(self, name: str) -> None:
        if not self.chroma_client:
            return
        try:
            self.chroma_client.delete([name])
        except Exception as exc:  # pragma: no cover - best effort removal
            logger.warning(
                "embedding_delete_failed",
                extra={
                    "tool": "technique_catalog",
                    "technique": name,
                    "error": str(exc),
                },
            )

    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
            return []

    def _write_dataset(self, entries: Iterable[dict[str, Any]]) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("w", encoding="utf-8") as handle:
            json.dump(list(entries), handle, ensure_ascii=False, indent=2)

    def _normalize_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(entry, dict):
            raise ValueError("Each technique entry must be a JSON object.")

        name = entry.get("name")
        description = entry.get("description")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Technique entries require a non-empty 'name'.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Technique entries require a non-empty 'description'.")

        normalized = {
            "name": name.strip(),
            "description": description.strip(),
            "origin_year": self._coerce_int(entry.get("origin_year")),
            "creator": self._normalize_optional(entry.get("creator")),
            "category": self._normalize_optional(entry.get("category")),
            "core_principles": self._normalize_optional(entry.get("core_principles")),
        }
        return normalized

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return int(str(value))

    @staticmethod
    def _normalize_optional(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
