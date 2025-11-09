from pathlib import Path
import sys
import types
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_litellm_stub() -> None:
    if "litellm" in sys.modules:
        return

    stub = types.ModuleType("litellm")

    def _raise_completion(*_: Any, **__: Any) -> dict[str, Any]:  # pragma: no cover
        raise RuntimeError("litellm completion should not be invoked in tests")

    def _raise_embedding(*_: Any, **__: Any) -> dict[str, Any]:  # pragma: no cover
        raise RuntimeError("litellm embedding should not be invoked in tests")

    stub.completion = _raise_completion
    stub.embedding = _raise_embedding
    stub.drop_params = True
    sys.modules["litellm"] = stub


_install_litellm_stub()
