from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.services.comparison_service import ComparisonResult
from src.services.simulation_service import SimulationResult
from src.workflows.compare_candidates import CompareCandidatesWorkflow
from src.workflows.config_update import ConfigUpdateWorkflow
from src.workflows.detect_technique import DetectTechniqueWorkflow
from src.workflows.feedback_loop import FeedbackWorkflow
from src.workflows.generate_plan import GeneratePlanWorkflow
from src.workflows.simulate_technique import SimulateTechniqueWorkflow


class StubSelector:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def recommend(self, description: str) -> dict[str, Any]:
        self.requested.append(description)
        return {"recommendation": {"suggested_technique": "Decisional Balance"}}


class StubPlanGenerator:
    def __init__(self) -> None:
        self.summaries: list[str] = []

    def generate(self, technique_summary: str) -> dict[str, Any]:
        self.summaries.append(technique_summary)
        return {"workflow": "summarize_result", "plan": ["Step 1"]}


class StubSimulationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def simulate(
        self,
        recommendation: dict[str, Any],
        *,
        problem_description: str | None,
        scenario: str | None,
        preference_summary: str | None,
    ) -> SimulationResult:
        self.calls.append(
            {
                "recommendation": recommendation,
                "problem_description": problem_description,
                "scenario": scenario,
                "preference_summary": preference_summary,
            }
        )
        return SimulationResult(
            simulation_overview="Walkthrough",
            scenario_variations=[{"name": "Best case", "outcome": "Success", "guidance": "Stay on plan"}],
            cautions=["Time pressure"],
            recommended_follow_up=["Review"],
            raw_response="{}",
        )


class StubComparisonService:
    def compare(
        self,
        recommendation: dict[str, Any],
        matches: list[dict[str, Any]],
        *,
        focus: str | None,
        preference_summary: str | None,
    ) -> ComparisonResult:
        return ComparisonResult(
            current_recommendation=recommendation.get("suggested_technique"),
            best_alternative="Six Thinking Hats",
            comparison_points=[{"technique": "Decisional Balance", "strengths": "Structured", "risks": "Slow", "best_for": "Trade-offs"}],
            decision_guidance=["Use hats for creativity"],
            confidence_notes="High",
            raw_response="{}",
        )


class StubFeedbackService:
    def __init__(self) -> None:
        self.recorded: list[dict[str, Any]] = []

    def record_feedback(
        self,
        *,
        workflow: str,
        message: str,
        rating: int | None,
        technique: str | None,
        category: str | None,
    ) -> None:
        self.recorded.append(
            {
                "workflow": workflow,
                "message": message,
                "rating": rating,
                "technique": technique,
                "category": category,
            }
        )

    def summarize_feedback(self) -> dict[str, str]:
        return {"workflow": "feedback_loop", "summary": "None"}


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_detect_technique_workflow_runs_selector() -> None:
    selector = StubSelector()
    workflow = DetectTechniqueWorkflow(selector=selector)

    result = workflow.run({"problem_description": "Need structure"})

    assert selector.requested == ["Need structure"]
    assert result["recommendation"]["suggested_technique"] == "Decisional Balance"


def test_detect_technique_workflow_requires_problem_description() -> None:
    workflow = DetectTechniqueWorkflow(selector=StubSelector())
    with pytest.raises(ValueError):
        workflow.run({})


def test_generate_plan_workflow_invokes_generator() -> None:
    generator = StubPlanGenerator()
    workflow = GeneratePlanWorkflow(plan_generator=generator)

    result = workflow.run({"technique_summary": "Summary"})

    assert generator.summaries == ["Summary"]
    assert result["plan"] == ["Step 1"]


def test_generate_plan_workflow_requires_summary() -> None:
    workflow = GeneratePlanWorkflow(plan_generator=StubPlanGenerator())
    with pytest.raises(ValueError):
        workflow.run({})


def test_simulate_technique_workflow_converts_result() -> None:
    service = StubSimulationService()
    workflow = SimulateTechniqueWorkflow(simulation_service=service)

    result = workflow.run(
        {
            "recommendation": {"suggested_technique": "Decisional Balance"},
            "problem_description": "Need structure",
            "scenario": "Negotiation",
            "preference_summary": "Prefers structure",
        }
    )

    assert service.calls
    assert result["simulation"]["simulation_overview"] == "Walkthrough"


def test_simulate_technique_requires_recommendation() -> None:
    workflow = SimulateTechniqueWorkflow(simulation_service=StubSimulationService())
    with pytest.raises(ValueError):
        workflow.run({})


def test_compare_candidates_workflow_returns_dict() -> None:
    workflow = CompareCandidatesWorkflow(comparison_service=StubComparisonService())
    comparison = workflow.run(
        {
            "recommendation": {"suggested_technique": "Decisional Balance"},
            "matches": [
                {
                    "metadata": {"name": "Decisional Balance", "category": "Decision Making"},
                    "score": 0.92,
                }
            ],
            "focus": "Decisional Balance",
            "preference_summary": "Prefers structure",
        }
    )

    assert comparison["workflow"] == "compare_candidates"
    assert comparison["comparison"]["best_alternative"] == "Six Thinking Hats"


def test_compare_candidates_requires_matches() -> None:
    workflow = CompareCandidatesWorkflow(comparison_service=StubComparisonService())
    with pytest.raises(ValueError):
        workflow.run({"recommendation": {}})


def test_feedback_workflow_dispatches_actions() -> None:
    service = StubFeedbackService()
    workflow = FeedbackWorkflow(feedback_service=service)

    workflow.run(
        {
            "action": "record",
            "message": "Great",
            "rating": 5,
            "technique": "Decisional Balance",
            "category": "Decision Making",
        }
    )
    assert service.recorded[0]["rating"] == 5

    summary = workflow.run({})
    assert summary["summary"] == "None"


def test_config_update_workflow_reads_configuration(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "settings.yaml",
        """
app:
  name: Test
logging:
  level: INFO
""".strip(),
    )
    _write_yaml(
        config_dir / "database.yaml",
        """
database:
  sqlite_path: ":memory:"
""".strip(),
    )
    _write_yaml(
        config_dir / "models.yaml",
        """
defaults:
  temperature: 0.3
  provider: openai
workflows:
  detect_technique:
    model: gpt-4.1
embeddings:
  model: text-embedding-3-large
  provider: openai
""".strip(),
    )
    _write_yaml(
        config_dir / "providers.yaml",
        """
providers:
  openai:
    api_key_env: OPENAI_API_KEY
""".strip(),
    )

    workflow = ConfigUpdateWorkflow(config_path=config_dir)
    snapshot = workflow.run({})

    assert snapshot["app"]["name"] == "Test"
    assert snapshot["database"]["sqlite_path"] == ":memory:"
    assert snapshot["embeddings"]["model"] == "text-embedding-3-large"
    assert "detect_technique" in snapshot["workflows"]
