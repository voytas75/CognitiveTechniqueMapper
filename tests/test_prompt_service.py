from pathlib import Path

import pytest

from src.services.prompt_service import PromptService


def test_prompt_service_loads_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    registry = prompts_dir / "registry.yaml"
    prompt_file = prompts_dir / "sample.txt"
    prompt_file.write_text("Sample prompt", encoding="utf-8")
    registry.write_text("sample: sample.txt\n", encoding="utf-8")

    monkeypatch.setenv("CTM_PROMPTS_PATH", str(prompts_dir))

    service = PromptService()
    assert "sample" in service.registry
    assert service.get_prompt("sample") == "Sample prompt"


def test_prompt_service_resolves_prefixed_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "project"
    prompts_dir = base / "prompts"
    prompts_dir.mkdir(parents=True)
    registry = prompts_dir / "registry.yaml"
    prompt_file = prompts_dir / "sample.txt"
    prompt_file.write_text("Sample prompt", encoding="utf-8")
    registry.write_text("sample: prompts/sample.txt\n", encoding="utf-8")

    monkeypatch.setenv("CTM_PROMPTS_PATH", str(prompts_dir))
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    service = PromptService()
    assert service.get_prompt("sample") == "Sample prompt"


def test_prompt_service_handles_empty_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "registry.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("CTM_PROMPTS_PATH", str(prompts_dir))

    service = PromptService()
    assert service.registry == {}


def test_prompt_service_rejects_non_mapping_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "registry.yaml").write_text("- sample\n", encoding="utf-8")
    monkeypatch.setenv("CTM_PROMPTS_PATH", str(prompts_dir))

    with pytest.raises(ValueError):
        PromptService()


def test_prompt_service_rejects_non_string_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "registry.yaml").write_text("sample: []\n", encoding="utf-8")
    monkeypatch.setenv("CTM_PROMPTS_PATH", str(prompts_dir))

    with pytest.raises(ValueError):
        PromptService()
