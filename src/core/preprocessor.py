from __future__ import annotations

import re
from typing import Protocol


class TextPreprocessor(Protocol):
    def normalize(self, text: str) -> str:
        ...


class ProblemPreprocessor:
    """Performs light normalization on user problem descriptions."""

    _whitespace_regex = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        cleaned = text.strip().lower()
        cleaned = self._whitespace_regex.sub(" ", cleaned)
        return cleaned
