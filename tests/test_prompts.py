from pathlib import Path

import yaml


def test_prompt_registry_paths_exist():
    registry_path = Path("prompts/registry.yaml")
    assert registry_path.exists()

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for prompt_name, prompt_path in registry.items():
        path = Path(prompt_path)
        assert path.exists(), f"Prompt file missing for {prompt_name}"
