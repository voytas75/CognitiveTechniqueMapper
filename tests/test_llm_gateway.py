from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.llm_gateway import LLMGateway
from src.services.config_service import ConfigService


def _write_llm_config(base_dir: Path) -> None:
    (base_dir / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (base_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (base_dir / "models.yaml").write_text(
        (
            "workflows:\n"
            "  detect_technique:\n"
            "    model: 'azure/gpt-4.1'\n"
            "    temperature: 0.2\n"
            "defaults: {provider: azure_openai}\n"
            "embeddings: {model: azure/UDTEMBED3L}\n"
        ),
        encoding="utf-8",
    )
    (base_dir / "providers.yaml").write_text(
        (
            "providers:\n"
            "  azure_openai:\n"
            "    api_base: 'https://azure.example.com'\n"
            "    api_version: '2024-05-01-preview'\n"
            "    api_key_env: 'AZURE_KEY'\n"
            "    litellm_provider: 'azure'\n"
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def llm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigService:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_llm_config(config_dir)
    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    monkeypatch.setenv("AZURE_KEY", "secret")
    return ConfigService(config_path=config_dir)


def test_llm_gateway_invokes_with_timeout(
    llm_config: ConfigService, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: Dict[str, Any] = {}

    def fake_completion(*, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("src.core.llm_gateway.completion", fake_completion)

    gateway = LLMGateway(config_service=llm_config)
    gateway._retry = Retrying(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )

    result = gateway.invoke("detect_technique", "hello world")

    assert result == "ok"
    assert captured["kwargs"]["timeout"] == LLMGateway.DEFAULT_TIMEOUT_SECONDS
    assert captured["messages"][0]["content"] == "hello world"


def test_llm_gateway_retries_on_failure(
    llm_config: ConfigService, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"count": 0}

    def flaky_completion(*, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr("src.core.llm_gateway.completion", flaky_completion)

    gateway = LLMGateway(config_service=llm_config)
    gateway._retry = Retrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )

    result = gateway.invoke("detect_technique", "prompt")

    assert result == "recovered"
    assert attempts["count"] == 3


def test_llm_gateway_raises_after_retry_exhaustion(
    llm_config: ConfigService, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"count": 0}

    def failing_completion(*, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        attempts["count"] += 1
        raise RuntimeError("persistent failure")

    monkeypatch.setattr("src.core.llm_gateway.completion", failing_completion)

    gateway = LLMGateway(config_service=llm_config)
    gateway._retry = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0, min=0, max=0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )

    with pytest.raises(LLMGateway.LLMInvocationError):
        gateway.invoke("detect_technique", "prompt")

    assert attempts["count"] == 3
