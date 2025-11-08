from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    import litellm
    from litellm import completion
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise RuntimeError("litellm is required for LLMGateway. Install via `pip install litellm`.") from exc

litellm.drop_params = True

from ..services.config_service import ConfigService, WorkflowModelConfig

logger = logging.getLogger(__name__)


class LLMGateway:
    """Central point for orchestrating LLM calls."""

    def __init__(self, config_service: ConfigService) -> None:
        self._config_service = config_service

    def invoke(
        self,
        workflow: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        config = self._config_service.get_workflow_model_config(workflow)
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_params(config, kwargs)

        try:
            logger.debug("Invoking workflow=%s model=%s", workflow, params.get("model"))
            response = completion(messages=messages, **params)
        except Exception as exc:  # pragma: no cover - network/credential issues
            logger.error("LLM invocation failed for %s: %s", workflow, exc)
            raise RuntimeError(f"LLM invocation failed for workflow '{workflow}': {exc}") from exc
        return response["choices"][0]["message"]["content"]

    def _build_messages(self, prompt: str, system_prompt: str | None) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_params(self, config: WorkflowModelConfig, overrides: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model": config.model}
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens

        provider_name = config.provider
        if provider_name:
            provider_config = dict(self._config_service.providers.get(provider_name, {}))
            api_key_env = provider_config.pop("api_key_env", None)
            params.update(
                {
                    key: value
                    for key, value in provider_config.items()
                    if key not in {"api_key_env"}
                }
            )
            if api_key_env:
                from os import environ

                api_key = environ.get(api_key_env)
                if not api_key:
                    message = (
                        f"Environment variable '{api_key_env}' required for provider '{provider_name}'."
                    )
                    logger.error(message)
                    raise RuntimeError(message)
                params.setdefault("api_key", api_key)

        params.update(overrides)
        return params
