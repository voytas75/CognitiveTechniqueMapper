"""Tests for the interactive question flow."""

from __future__ import annotations

from typing import Iterator, List

import pytest
from rich.console import Console

import src.services.interactive_question_flow as flow_module
from src.services.decision_tree_flow import DecisionTreeDefinition
from src.services.interactive_question_flow import (
    InvalidDecisionTreeError,
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


def test_create_interactive_flow_rejects_root_without_a_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_definition = DecisionTreeDefinition(root="missing", nodes={}, techniques={})
    monkeypatch.setattr(
        flow_module, "get_decision_tree_definition", lambda: invalid_definition
    )

    with pytest.raises(InvalidDecisionTreeError, match="root"):
        create_interactive_flow(StubConsole([]))


def test_interactive_flow_retries_numeric_input_and_renders_visualization() -> None:
    console = StubConsole(["invalid", "3", "1"])
    flow = create_interactive_flow(console, show_visualization=True)

    technique_id, technique_label = flow.run()

    assert (technique_id, technique_label) == ("chain_of_thought", "CoT")
    output = console.export_text()
    assert "Invalid answer" in output
    assert "Decision Tree" in output


def test_interactive_flow_handles_clarification_action() -> None:
    console = StubConsole(["unknown"])

    assert create_interactive_flow(console).run() == (None, None)
    assert "Request clarification" in console.export_text()
