"""Interactive decision-tree question flow service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rich.console import Console
from rich.panel import Panel

from src.services.decision_tree_flow import (
    DecisionTreeDefinition,
    DecisionTreeVisualizer,
    get_decision_tree_definition,
    humanize_identifier,
)


class InvalidDecisionTreeError(RuntimeError):
    """Raised when the decision tree definition is invalid."""


class DecisionTreeTraversalError(RuntimeError):
    """Raised when traversal encounters an invalid path."""


def normalize_response(value: str | None) -> str:
    """Normalize user input to a lowercase token.

    Args:
        value (str | None): Raw user input.

    Returns:
        str: Normalized representation suitable for branch matching.
    """

    if value is None:
        return ""
    cleaned = value.strip().lower()
    replacements = {"y": "yes", "n": "no"}
    return replacements.get(cleaned, cleaned)


@dataclass
class InteractiveQuestionFlow:
    """Interactive traversal of the decision tree."""

    definition: DecisionTreeDefinition
    console: Console
    show_visualization: bool = False

    def run(self) -> tuple[str | None, str | None]:
        """Traverse decision tree based on user input.

        Returns:
            tuple[str | None, str | None]: Selected technique identifier and label.
        """

        current_node_id = self.definition.root
        while True:
            node = self.definition.get_node(current_node_id)
            if node is None:
                if self.definition.is_technique_key(current_node_id):
                    label = self.definition.technique_label(current_node_id)
                    return current_node_id, label
                raise DecisionTreeTraversalError(f"Unknown node: {current_node_id}")

            if node.action and not node.branches:
                self.console.print(Panel(node.action, title=humanize_identifier(node.name)))
                return None, None

            answer = self._prompt_node(node.name, node.question, node.branches)
            next_node_id = self._resolve_branch(answer, node.branches)
            if not next_node_id:
                continue
            if self.definition.is_technique_key(next_node_id):
                label = self.definition.technique_label(next_node_id)
                self.console.print(
                    Panel(
                        f"Technique selected: {humanize_identifier(next_node_id)} ({label})",
                        title="Interactive Flow",
                        style="green",
                    )
                )
                if self.show_visualization:
                    self._render_visualization()
                return next_node_id, label
            current_node_id = next_node_id

    def _prompt_node(
        self,
        node_name: str,
        question: str | None,
        branches: dict[str, str],
    ) -> str:
        """Prompt the user for an answer.

        Args:
            node_name (str): Current node identifier.
            question (str | None): Question text for the node.
            branches (dict[str, str]): Available branch identifiers.

        Returns:
            str: Raw user response.
        """

        prompt_label = question or humanize_identifier(node_name)
        options = self._format_options(branches.keys())
        if options:
            self.console.print(
                Panel(
                    "\n".join(options),
                    title=f"Options for {humanize_identifier(node_name)}",
                    subtitle="Enter value or number",
                )
            )
        return self.console.input(f"[bold cyan]{prompt_label}:[/] ")

    def _format_options(self, options: Iterable[str]) -> list[str]:
        """Format branch options for display.

        Args:
            options (Iterable[str]): Iterable of branch keys.

        Returns:
            list[str]: Numbered option labels.
        """

        formatted: list[str] = []
        for index, option in enumerate(options, start=1):
            formatted.append(f"{index}. {humanize_identifier(option)} ({option})")
        return formatted

    def _resolve_branch(self, answer: str, branches: dict[str, str]) -> str | None:
        """Resolve the branch identifier based on user input.

        Args:
            answer (str): Raw user response.
            branches (dict[str, str]): Mapping of answers to branch IDs.

        Returns:
            str | None: Next node identifier or ``None`` if invalid.
        """

        normalized = normalize_response(answer)
        if normalized in branches:
            return branches[normalized]
        if normalized.isdigit():
            index = int(normalized) - 1
            branch_keys = list(branches.keys())
            if 0 <= index < len(branch_keys):
                return branches[branch_keys[index]]
        valid_options = ", ".join(branches.keys())
        self.console.print(
            Panel(
                f"Unrecognized answer: {answer}. Valid options: {valid_options}",
                title="Invalid answer",
                style="red",
            )
        )
        return None

    def _render_visualization(self) -> None:
        """Render the decision tree visualization."""

        visualizer = DecisionTreeVisualizer(self.definition)
        self.console.print(
            Panel(
                visualizer.render(),
                title="Decision Tree",
                subtitle="Interactive Question Flow",
                expand=False,
            )
        )


def create_interactive_flow(
    console: Console, *, show_visualization: bool = False
) -> InteractiveQuestionFlow:
    """Factory helper for the interactive question flow.

    Args:
        console (Console): Rich console used for I/O.
        show_visualization (bool): Whether to render the full tree after completion.

    Returns:
        InteractiveQuestionFlow: Configured flow instance.
    """

    definition = get_decision_tree_definition()
    if not definition.root:
        raise InvalidDecisionTreeError("Decision tree root is undefined")
    return InteractiveQuestionFlow(
        definition=definition,
        console=console,
        show_visualization=show_visualization,
    )


__all__ = [
    "DecisionTreeTraversalError",
    "InteractiveQuestionFlow",
    "InvalidDecisionTreeError",
    "create_interactive_flow",
    "normalize_response",
]

