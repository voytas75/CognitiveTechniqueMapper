"""Text preprocessing utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module-level and method docstrings.
"""

from __future__ import annotations

import re
from typing import Protocol


class TextPreprocessor(Protocol):
    """Protocol describing preprocessor capabilities."""

    def normalize(self, text: str) -> str:
        """Normalize the provided text into a canonical form.

        Args:
            text (str): Input string to normalize.

        Returns:
            str: Normalized text representation.
        """

        ...


class ProblemPreprocessor:
    """Performs light normalization on user problem descriptions."""

    _whitespace_regex = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        """Normalize whitespace and casing for a problem description.

        Args:
            text (str): Raw user-provided problem description.

        Returns:
            str: Lower-cased description with collapsed whitespace.
        """

        cleaned = text.strip().lower()
        cleaned = self._whitespace_regex.sub(" ", cleaned)
        return cleaned
