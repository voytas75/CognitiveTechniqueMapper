"""LLM gateway for workflow-specific completions.

Updates:
    v0.1.1 - 2025-11-09 - Resolve max_tokens from LiteLLM metadata when available.
    v0.1.0 - 2025-11-09 - Added Google-style docstrings and update metadata.
    v0.3.0 - 2025-05-09 - Added tenacity retries, timeouts, and structured errors.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, cast

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..services.config_service import ConfigService, WorkflowModelConfig

try:
    import litellm
    from litellm import completion as _litellm_completion  # pyright: ignore[reportUnknownVariableType]
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise RuntimeError(
        "litellm is required for LLMGateway. Install via `pip install litellm`."
    ) from exc
else:
    litellm.drop_params = True

if TYPE_CHECKING:
    from typing import Protocol

    class CompletionCallable(Protocol):
        def __call__(
            self,
            *,
            messages: List[Dict[str, str]],
            **kwargs: Any,
        ) -> Dict[str, Any]:
            ...
else:
    CompletionCallable = Callable[..., Any]

completion = _litellm_completion

logger = logging.getLogger(__name__)


class LLMGateway:
    """Central point for orchestrating LLM calls."""

    DEFAULT_TIMEOUT_SECONDS = 30

    class LLMInvocationError(RuntimeError):
        """Raised when an LLM invocation fails after retries."""

    def __init__(self, config_service: ConfigService) -> None:
        """Store configuration dependencies for LLM dispatch.

        Args:
            config_service (ConfigService): Loader providing workflow model configuration.
        """

        self._config_service = config_service
        self._retry = Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

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

        params.setdefault("timeout", self.DEFAULT_TIMEOUT_SECONDS)

        try:
            response = self._execute_with_retry(workflow, messages, params)
        except RetryError as retry_error:  # pragma: no cover - relied on tests
            last_exc = retry_error.last_attempt.exception()
            message = (
                f"LLM invocation failed for workflow '{workflow}' after retries: {last_exc}"
            )
            logger.error(message)
            raise self.LLMInvocationError(message) from last_exc
        except Exception as exc:  # pragma: no cover - network/credential issues
            message = f"LLM invocation failed for workflow '{workflow}': {exc}"
            logger.error(message)
            raise self.LLMInvocationError(message) from exc

        return self._extract_text_content(response)

    def _execute_with_retry(
        self,
        workflow: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        completion_fn = cast(CompletionCallable, completion)
        for attempt in self._retry:
            with attempt:
                logger.debug(
                    "Invoking workflow=%s model=%s attempt=%s",
                    workflow,
                    params.get("model"),
                    attempt.retry_state.attempt_number,
                )
                raw_response = completion_fn(messages=messages, **params)
                return self._normalise_response(raw_response)
        raise self.LLMInvocationError(
            f"LLM invocation failed for workflow '{workflow}' without response."
        )

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

        max_tokens = self._resolve_max_tokens(config)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        provider_name = config.provider
        if provider_name:
            provider_config = dict(self._config_service.providers.get(provider_name, {}))
            api_key_env = provider_config.pop("api_key_env", None)
            litellm_provider = provider_config.pop("litellm_provider", None)
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
            params.setdefault(
                "custom_llm_provider", litellm_provider or provider_name
            )

        params.update(overrides)
        return params

    def _extract_text_content(self, response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise self.LLMInvocationError(
                "LLM response did not include any choices."
            )
        mapping_choices = cast(Sequence[Mapping[str, object]], choices)
        first_choice = mapping_choices[0]
        message_value: object | None = first_choice.get("message")
        if not isinstance(message_value, dict):
            raise self.LLMInvocationError("LLM response missing message object.")
        message_mapping = cast(Mapping[str, Any], message_value)
        content = message_mapping.get("content")
        if not isinstance(content, str):
            raise self.LLMInvocationError("LLM response content is not textual.")
        return content

    def _normalise_response(self, raw_response: Any) -> Dict[str, Any]:
        if isinstance(raw_response, dict):
            return cast(Dict[str, Any], raw_response)
        dict_method = getattr(raw_response, "dict", None)
        if callable(dict_method):
            candidate = dict_method()
            if isinstance(candidate, dict):
                return cast(Dict[str, Any], candidate)
        model_dump = getattr(raw_response, "model_dump", None)
        if callable(model_dump):
            candidate = model_dump()
            if isinstance(candidate, dict):
                return cast(Dict[str, Any], candidate)
        raise self.LLMInvocationError(
            f"Unexpected LLM response type: {type(raw_response).__name__}"
        )

    def _resolve_max_tokens(self, config: WorkflowModelConfig) -> Optional[int]:
        """Determine the max_tokens parameter for the configured model.

        Args:
            config (WorkflowModelConfig): Workflow-specific configuration values.

        Returns:
            int | None: LiteLLM-suggested maximum tokens or configuration override.
        """

        if config.max_tokens is not None:
            return config.max_tokens

        suggested = self._suggested_max_tokens(config.model)
        if suggested is not None:
            logger.debug(
                "Resolved max_tokens=%s from LiteLLM metadata for model=%s",
                suggested,
                config.model,
            )
        return suggested

    def _suggested_max_tokens(self, model: str) -> Optional[int]:
        """Look up LiteLLM metadata for the provided model identifier.

        Args:
            model (str): Model identifier supplied to LiteLLM.

        Returns:
            int | None: Max token suggestion when available.
        """

        candidates = [model]
        if "/" in model:
            candidates.append(model.split("/", 1)[1])

        for name in candidates:
            suggested = self._lookup_litellm_max_tokens(name)
            if suggested is not None:
                return suggested
        return None

    def _lookup_litellm_max_tokens(self, model: str) -> Optional[int]:
        """Extract a max token value from LiteLLM metadata registries.

        Args:
            model (str): Model identifier to inspect.

        Returns:
            int | None: Max tokens when reported by LiteLLM.
        """

        for accessor in (
            getattr(litellm, "get_model_info", None),
            getattr(getattr(litellm, "utils", None), "get_model_info", None),
        ):
            if not callable(accessor):
                continue
            info = self._safe_model_info(accessor, model)
            if info is None:
                continue
            suggestion = self._extract_max_tokens(info)
            if suggestion is not None:
                return suggestion

        registry = getattr(litellm, "model_cost", None)
        if isinstance(registry, Mapping):
            cost_table = cast(Mapping[str, Mapping[str, Any]], registry)
            entry = cost_table.get(model)
            if entry is not None:
                suggestion = self._extract_max_tokens(entry)
                if suggestion is not None:
                    return suggestion

        return None

    @staticmethod
    def _extract_max_tokens(info: Mapping[str, Any]) -> Optional[int]:
        """Normalize max token values from LiteLLM metadata responses."""
        for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
            value = info.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        return None

    def _safe_model_info(
        self, accessor: Callable[..., Any], model: str
    ) -> Mapping[str, Any] | None:
        try:
            result = accessor(model)
        except TypeError:
            try:
                result = accessor(model=model)
            except Exception as exc:  # pragma: no cover - noisy debug path
                logger.debug("LiteLLM get_model_info(%s) failed: %s", model, exc)
                return None
        except Exception as exc:  # pragma: no cover - noisy debug path
            logger.debug("LiteLLM get_model_info(%s) failed: %s", model, exc)
            return None

        if isinstance(result, Mapping):
            return cast(Mapping[str, Any], result)
        dict_method = getattr(result, "dict", None)
        if callable(dict_method):
            candidate = dict_method()
            if isinstance(candidate, Mapping):
                return cast(Mapping[str, Any], candidate)
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            candidate = model_dump()
            if isinstance(candidate, Mapping):
                return cast(Mapping[str, Any], candidate)
        return None
