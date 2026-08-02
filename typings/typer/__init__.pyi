"""Minimal local typing surface for the Typer APIs used by CTM.

Typer 0.23.1 exposes Click generic parameters as Unknown to Pyright strict mode.
This stub is intentionally limited to CTM's imported public surface and has no
runtime effect.
"""

from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def Option(default: Any = ..., *param_decls: str, **kwargs: Any) -> Any: ...
def Argument(default: Any = ..., *param_decls: str, **kwargs: Any) -> Any: ...
def confirm(text: str, *, abort: bool = False, default: bool = False) -> bool: ...
def prompt(text: str, default: Any = ..., **kwargs: Any) -> Any: ...


class Exit(Exception):
    def __init__(self, code: int | None = ...) -> None: ...


class BadParameter(Exception):
    def __init__(self, message: str, **kwargs: Any) -> None: ...


class Context:
    invoked_subcommand: str | None


class Typer:
    def __init__(self, **kwargs: Any) -> None: ...
    def command(self, *args: Any, **kwargs: Any) -> Callable[[_F], _F]: ...
    def callback(self, *args: Any, **kwargs: Any) -> Callable[[_F], _F]: ...
    def add_typer(self, typer_instance: Typer, *args: Any, **kwargs: Any) -> None: ...
