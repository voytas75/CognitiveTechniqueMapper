"""Utilities for assembling CLI reports."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.cli.state import AppState


def build_report_payload(state: AppState) -> Dict[str, Any]:
    """Construct a report payload from the current CLI state."""

    recommendation: Optional[Dict[str, Any]] = None
    if state.last_recommendation:
        recommendation = dict(
            state.last_recommendation.get("recommendation") or {}
        )
        plan = state.last_recommendation.get("plan")
        if recommendation is not None and plan is not None:
            recommendation.setdefault("plan", plan)

    return {
        "problem_description": state.problem_description,
        "recommendation": recommendation,
        "explanation": state.last_explanation,
        "simulation": state.last_simulation,
        "comparison": state.last_comparison,
    }


def render_report_markdown(payload: Dict[str, Any]) -> str:
    """Render a markdown report from report payload."""

    sections: list[str] = []
    problem = payload.get("problem_description") or "N/A"
    sections.append(f"# Cognitive Technique Report\n\n## Problem Statement\n\n{problem}\n")

    recommendation = payload.get("recommendation") or {}
    rec_body = _render_recommendation(recommendation)
    if rec_body:
        sections.append(rec_body)

    explanation = payload.get("explanation") or {}
    exp_body = _render_explanation(explanation)
    if exp_body:
        sections.append(exp_body)

    simulation = payload.get("simulation") or {}
    sim_body = _render_simulation(simulation)
    if sim_body:
        sections.append(sim_body)

    comparison = payload.get("comparison") or {}
    cmp_body = _render_comparison(comparison)
    if cmp_body:
        sections.append(cmp_body)

    sections.append("## Appendix\n\n```json\n" + json.dumps(payload, indent=2) + "\n```")
    return "\n\n".join(sections)


def _render_recommendation(recommendation: Dict[str, Any]) -> Optional[str]:
    if not recommendation:
        return None
    suggested = recommendation.get("suggested_technique") or "Unknown"
    why = recommendation.get("why_it_fits") or "No rationale provided."
    steps = recommendation.get("steps") or []
    md = ["## Recommendation", f"**Technique:** {suggested}", f"**Why it fits:** {why}"]
    if steps:
        md.append("**Suggested Steps:**")
        for idx, step in enumerate(steps, start=1):
            md.append(f"{idx}. {step}")
    plan = recommendation.get("plan")
    if plan:
        md.append("\n**Plan Output:**\n")
        if isinstance(plan, dict):
            md.append("```json\n" + json.dumps(plan, indent=2) + "\n```")
        else:
            md.append(str(plan))
    return "\n\n".join(md)


def _render_explanation(explanation: Dict[str, Any]) -> Optional[str]:
    if not explanation:
        return None
    md = ["## Explanation"]
    if explanation.get("overview"):
        md.append(explanation["overview"])
    for key in ("key_factors", "risks", "next_steps"):
        values = explanation.get(key) or []
        if values:
            header = key.replace("_", " ").title()
            md.append(f"**{header}:**")
            for idx, entry in enumerate(values, start=1):
                md.append(f"{idx}. {entry}")
    return "\n\n".join(md)


def _render_simulation(simulation: Dict[str, Any]) -> Optional[str]:
    if not simulation:
        return None
    md = ["## Simulation"]
    overview = simulation.get("simulation_overview")
    if overview:
        md.append(overview)
    variations = simulation.get("scenario_variations") or []
    if variations:
        md.append("**Scenario Variations:**")
        for entry in variations:
            name = entry.get("name") or "Scenario"
            outcome = entry.get("outcome") or ""
            guidance = entry.get("guidance") or ""
            line = f"- {name}: {outcome}"
            if guidance:
                line += f" (Guidance: {guidance})"
            md.append(line)
    cautions = simulation.get("cautions") or []
    if cautions:
        md.append("**Cautions:**")
        for idx, caution in enumerate(cautions, start=1):
            md.append(f"{idx}. {caution}")
    follow_up = simulation.get("recommended_follow_up") or []
    if follow_up:
        md.append("**Recommended Follow-up:**")
        for idx, action in enumerate(follow_up, start=1):
            md.append(f"{idx}. {action}")
    return "\n\n".join(md)


def _render_comparison(comparison: Dict[str, Any]) -> Optional[str]:
    if not comparison:
        return None
    md = ["## Comparison"]
    current = comparison.get("current_recommendation") or "Unknown"
    md.append(f"**Current Recommendation:** {current}")
    alternative = comparison.get("best_alternative")
    if alternative:
        md.append(f"**Top Alternative:** {alternative}")
    points = comparison.get("comparison_points") or []
    if points:
        md.append("**Comparison Points:**")
        for point in points:
            technique = point.get("technique") or "Candidate"
            strengths = point.get("strengths") or ""
            risks = point.get("risks") or ""
            best_for = point.get("best_for") or ""
            md.append(f"- {technique}")
            if strengths:
                md.append(f"  - Strengths: {strengths}")
            if risks:
                md.append(f"  - Risks: {risks}")
            if best_for:
                md.append(f"  - Best for: {best_for}")
    guidance = comparison.get("decision_guidance") or []
    if guidance:
        md.append("**Decision Guidance:**")
        for idx, tip in enumerate(guidance, start=1):
            md.append(f"{idx}. {tip}")
    confidence = comparison.get("confidence_notes")
    if confidence:
        md.append(f"**Confidence Notes:** {confidence}")
    return "\n\n".join(md)


__all__ = ["build_report_payload", "render_report_markdown"]
