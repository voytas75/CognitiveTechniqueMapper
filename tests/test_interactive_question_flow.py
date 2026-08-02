"""Tests for the interactive question flow."""

from __future__ import annotations

from typing import Iterator, List

from rich.console import Console

from src.services.interactive_question_flow import (
    create_interactive_flow,
    normalize_response,
)


class StubConsole(Console):
    """Console that feeds scripted inputs and captures prints."""

    def __init__(self, inputs: List[str]) -> None:
        super().__init__(record=True, width=80)
        self._inputs: Iterator[str] = iter(inputs)

    def input(self, prompt: str = "", *, markup: bool = True) -> str:  # type: ignore[override]
        try:
            return next(self._inputs)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError("No more scripted inputs") from exc


def test_normalize_response_aliases() -> None:
    assert normalize_response("Y") == "yes"
    assert normalize_response("n") == "no"
    assert normalize_response("Maybe") == "maybe"


def test_interactive_flow_reaches_leaf(monkeypatch: None) -> None:
    console = StubConsole(
        [
            "reasoning",
            "yes",
        ]
    )
    flow = create_interactive_flow(console, show_visualization=False)
    technique_id, technique_label = flow.run()
    assert technique_id == "chain_of_thought"
    assert technique_label == flow.definition.technique_label("chain_of_thought")
