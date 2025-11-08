"""Technique utility helpers.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and Google-style function docstring.
"""

from __future__ import annotations

from typing import Mapping


def compose_embedding_text(item: Mapping[str, str]) -> str:
    """Build a rich text representation used for embedding techniques.

    Args:
        item (Mapping[str, str]): Technique metadata dictionary.

    Returns:
        str: Concatenated text used when generating embedding vectors.
    """

    description = item.get("description", "")
    principles = item.get("core_principles", "")
    category = item.get("category", "")
    return " \n".join(filter(None, [description, principles, category]))
