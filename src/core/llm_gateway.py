"""LLM gateway for workflow-specific completions.

Updates:
    v0.1.0 - 2025-11-09 - Added Google-style docstrings and update metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    import litellm
    from litellm import completion
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise RuntimeError(
        "litellm is required for LLMGateway. Install via `pip install litellm`."
    ) from exc

litellm.drop_params = True

from ..services.config_service import ConfigService, WorkflowModelConfig

logger = logging.getLogger(__name__)


class LLMGateway:
    """Central point for orchestrating LLM calls."""

    def __init__(self, config_service: ConfigService) -> None:
        """Store configuration dependencies for LLM dispatch.

        Args:
            config_service (ConfigService): Loader providing workflow model configuration.
        """

        self._config_service = config_service

    def invoke(
        self,
        workflow: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Invoke the configured LLM workflow and return the response text.

        Args:
            workflow (str): Name of the workflow to execute.
            prompt (str): User-facing prompt content.
            system_prompt (str | None): Optional system guidance passed to the model.
            **kwargs: Provider-specific overrides for the request.

        Returns:
            str: Content string returned by the LLM provider.

        Raises:
            RuntimeError: If the provider invocation fails.
        """

        config = self._config_service.get_workflow_model_config(workflow)
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_params(config, kwargs)

        try:
            logger.debug("Invoking workflow=%s model=%s", workflow, params.get("model"))
            response = completion(messages=messages, **params)
        except Exception as exc:  # pragma: no cover - network/credential issues
            logger.error("LLM invocation failed for %s: %s", workflow, exc)
            raise RuntimeError(
                f"LLM invocation failed for workflow '{workflow}': {exc}"
            ) from exc
        return response["choices"][0]["message"]["content"]

    def _build_messages(
        self, prompt: str, system_prompt: str | None
    ) -> List[Dict[str, str]]:
        """Construct the chat completion message payload.

        Args:
            prompt (str): User prompt content for the LLM.
            system_prompt (str | None): Optional system-level instructions.

        Returns:
            list[dict[str, str]]: Message records following the chat schema.
        """

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_params(
        self, config: WorkflowModelConfig, overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the request parameters merged with overrides and provider settings.

        Args:
            config (WorkflowModelConfig): Workflow-specific configuration values.
            overrides (dict[str, Any]): Parameters supplied by the caller.

        Returns:
            dict[str, Any]: Completed parameter payload for the provider call.

        Raises:
            RuntimeError: If a configured provider is missing the required API key.
        """

        params: Dict[str, Any] = {"model": config.model}
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens

        provider_name = config.provider
        if provider_name:
            provider_config = dict(
                self._config_service.providers.get(provider_name, {})
            )
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
                    message = f"Environment variable '{api_key_env}' required for provider '{provider_name}'."
                    logger.error(message)
                    raise RuntimeError(message)
                params.setdefault("api_key", api_key)

        params.update(overrides)
        return params
