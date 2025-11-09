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
