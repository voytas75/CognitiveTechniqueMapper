from __future__ import annotations

from typing import Mapping


def compose_embedding_text(item: Mapping[str, str]) -> str:
    """Builds a rich text representation used for embedding techniques."""
    description = item.get("description", "")
    principles = item.get("core_principles", "")
    category = item.get("category", "")
    return " \n".join(filter(None, [description, principles, category]))
