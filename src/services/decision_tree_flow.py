"""Decision tree utilities for the interactive question flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

DECISION_TREE_DATA: Dict[str, Any] = {
    "decision_tree": {
        "root": "task_type",
        "nodes": {
            "task_type": {
                "question": "Identify task type",
                "branches": {
                    "knowledge_retrieval": "knowledge_branch",
                    "text_processing": "text_branch",
                    "reasoning": "reasoning_branch",
                    "code_generation": "code_branch",
                    "creative_generation": "creative_branch",
                    "unknown": "clarification",
                },
            },
            "knowledge_branch": {
                "question": "Is the answer short, factual, and atomic?",
                "branches": {"yes": "direct_fact_retrieval", "no": "knowledge_complex"},
            },
            "knowledge_complex": {
                "question": "Is the question complex and multi-layered?",
                "branches": {
                    "yes": "structured_decomposition",
                    "no": "knowledge_uncertainty",
                },
            },
            "knowledge_uncertainty": {
                "question": "Is uncertainty acceptable?",
                "branches": {
                    "yes": "self_check_retrieval",
                    "no": "constrained_knowledge_answering",
                },
            },
            "text_branch": {
                "question": "Do you need clean field/entity extraction?",
                "branches": {
                    "yes": "schema_guided_extraction",
                    "no": "text_summarization_branch",
                },
            },
            "text_summarization_branch": {
                "question": "Do you need a shorter version preserving meaning?",
                "branches": {
                    "yes": "abstractive_summarization",
                    "no": "text_classification_branch",
                },
            },
            "text_classification_branch": {
                "question": "Do you need comparison or classification?",
                "branches": {
                    "yes": "contrastive_classification",
                    "no": "context_preserving_rewrite",
                },
            },
            "reasoning_branch": {
                "question": "Is it a multi-step logical or numeric task?",
                "branches": {"yes": "chain_of_thought", "no": "reasoning_ambiguity"},
            },
            "reasoning_ambiguity": {
                "question": "Does the task require exploring multiple hypotheses?",
                "branches": {
                    "yes": "tree_of_thought",
                    "no": "reasoning_precision",
                },
            },
            "reasoning_precision": {
                "question": "Is precision critical?",
                "branches": {
                    "yes": "self_consistency_reasoning",
                    "no": "direct_structured_reasoning",
                },
            },
            "code_branch": {
                "question": "Is it a small direct function generation?",
                "branches": {
                    "yes": "direct_code_generation",
                    "no": "code_decomposition",
                },
            },
            "code_decomposition": {
                "question": "Is the task large and requires decomposition?",
                "branches": {"yes": "plan_and_implement", "no": "code_error_free"},
            },
            "code_error_free": {
                "question": "Is an error-free result required?",
                "branches": {
                    "yes": "round_trip_validation",
                    "no": "explain_and_generate",
                },
            },
            "creative_branch": {
                "question": "Do you need variations or alternatives?",
                "branches": {
                    "yes": "divergent_expansion",
                    "no": "creative_constraints",
                },
            },
            "creative_constraints": {
                "question": "Are there strict constraints?",
                "branches": {
                    "yes": "constrained_creative_generation",
                    "no": "unconstrained_creative_generation",
                },
            },
            "clarification": {
                "action": "Request clarification about the task type.",
                "branches": {},
            },
        },
        "techniques": {
            "direct_fact_retrieval": "DFR",
            "structured_decomposition": "SD",
            "self_check_retrieval": "SCR",
            "constrained_knowledge_answering": "CKA",
            "schema_guided_extraction": "SGE",
            "abstractive_summarization": "AS",
            "contrastive_classification": "CC",
            "context_preserving_rewrite": "CPR",
            "chain_of_thought": "CoT",
            "tree_of_thought": "ToT",
            "self_consistency_reasoning": "SCR2",
            "direct_structured_reasoning": "DSR",
            "direct_code_generation": "DCG",
            "plan_and_implement": "PAI",
            "round_trip_validation": "RTV",
            "explain_and_generate": "EG",
            "divergent_expansion": "DE",
            "constrained_creative_generation": "CCG",
            "unconstrained_creative_generation": "UCG",
        },
    }
}


@dataclass(frozen=True)
class DecisionTreeNode:
    """Single decision node in the interactive tree."""

    name: str
    question: str | None = None
    branches: Dict[str, str] = field(default_factory=dict)
    action: str | None = None


@dataclass(frozen=True)
class DecisionTreeDefinition:
    """Definition of the decision tree used by the interactive flow."""

    root: str
    nodes: Dict[str, DecisionTreeNode]
    techniques: Dict[str, str]

    def get_node(self, node_id: str) -> DecisionTreeNode | None:
        """Return the node associated with ``node_id``.

        Args:
            node_id (str): Identifier of the node to retrieve.

        Returns:
            DecisionTreeNode | None: Node definition if it exists.
        """

        return self.nodes.get(node_id)

    def is_technique_key(self, identifier: str) -> bool:
        """Check whether ``identifier`` maps to a technique leaf.

        Args:
            identifier (str): Potential technique identifier.

        Returns:
            bool: ``True`` if the identifier corresponds to a technique.
        """

        return identifier in self.techniques

    def technique_label(self, identifier: str) -> str:
        """Return the short label for ``identifier``.

        Args:
            identifier (str): Technique identifier.

        Returns:
            str: Abbreviated label for the technique.
        """

        return self.techniques[identifier]


def get_decision_tree_definition() -> DecisionTreeDefinition:
    """Return the predefined decision tree definition.

    Returns:
        DecisionTreeDefinition: Parsed tree containing nodes and techniques.
    """

    tree_data = DECISION_TREE_DATA["decision_tree"]
    nodes: Dict[str, DecisionTreeNode] = {}
    for node_id, payload in tree_data["nodes"].items():
        nodes[node_id] = DecisionTreeNode(
            name=node_id,
            question=payload.get("question"),
            branches=dict(payload.get("branches", {})),
            action=payload.get("action"),
        )
    return DecisionTreeDefinition(
        root=tree_data["root"],
        nodes=nodes,
        techniques=dict(tree_data["techniques"]),
    )


def humanize_identifier(value: str) -> str:
    """Convert a machine identifier into a human-readable title.

    Args:
        value (str): Identifier to convert.

    Returns:
        str: Human-readable version of the identifier.
    """

    if not value:
        return value
    cleaned = value.replace("_", " ")
    return cleaned.strip().title()


@dataclass
class DecisionTreeVisualizer:
    """Renderable text visualization of the decision tree."""

    definition: DecisionTreeDefinition

    def render(self) -> str:
        """Return an indented text representation of the tree.

        Returns:
            str: Multiline string representing the tree structure.
        """

        lines: List[str] = []
        self._render_node(self.definition.root, lines, 0)
        return "\n".join(lines)

    def _render_node(self, node_id: str, lines: List[str], depth: int) -> None:
        """Append visualization lines for ``node_id``.

        Args:
            node_id (str): Node identifier to render.
            lines (List[str]): Accumulator for output lines.
            depth (int): Current indentation depth.
        """

        node = self.definition.get_node(node_id)
        indent = "  " * depth
        if node is None:
            lines.append(f"{indent}- {humanize_identifier(node_id)} (leaf)")
            return
        label = node.question or node.action or humanize_identifier(node.name)
        lines.append(f"{indent}- {humanize_identifier(node.name)}: {label}")
        for branch, target in node.branches.items():
            branch_label = humanize_identifier(branch)
            lines.append(
                f"{indent}  [ {branch_label} ] -> {humanize_identifier(target)}"
            )
            if self.definition.get_node(target):
                self._render_node(target, lines, depth + 2)
            elif self.definition.is_technique_key(target):
                technique_label = humanize_identifier(target)
                code = self.definition.techniques[target]
                lines.append(f"{indent}    • Technique: {technique_label} ({code})")
