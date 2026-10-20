"""Validated lazy registries for ArityBench task families and methods."""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")

TASKS: dict[str, type] = {}
METHODS: dict[str, type] = {}
_BUILTINS_LOADED = False


def _validate_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} registry name must be a string")
    name = name.strip()
    if not name or not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid {kind} registry name {name!r}; use letters, digits, '_' or '-', "
            "starting with a letter"
        )
    return name


def _decorator(name: str, registry: dict[str, type], kind: str) -> Callable[[T], T]:
    name = _validate_name(name, kind)

    def register(cls: T) -> T:
        if not isinstance(cls, type):
            raise TypeError(f"@{kind} may decorate classes only")
        existing = registry.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{kind} {name!r} already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        registry[name] = cls
        cls.registry_name = name
        return cls

    return register


def task(name: str):
    """Register a task-family class under ``name``."""
    return _decorator(name, TASKS, "task")


def method(name: str):
    """Register a method class under ``name``."""
    return _decorator(name, METHODS, "method")


def load_builtin_registrations() -> None:
    """Import built-in task and method packages exactly once.

    Registration remains decorator-based, but callers no longer need to know
    which side-effect imports populate the registries.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    package = __package__
    if not package:
        raise RuntimeError("registry.py must be imported as part of a package")

    importlib.import_module(f"{package}.tasks")
    try:
        importlib.import_module(f"{package}.methods")
    except ModuleNotFoundError as exc:
        # A task-only installation is useful for target audits. Suppress only
        # the absence of the package itself, not errors inside an existing one.
        if exc.name != f"{package}.methods":
            raise

    _BUILTINS_LOADED = True


def get_task(name: str):
    load_builtin_registrations()
    name = _validate_name(name, "task")
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name]()


def get_method(name: str):
    load_builtin_registrations()
    name = _validate_name(name, "method")
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)}")
    return METHODS[name]()


def available_tasks() -> tuple[str, ...]:
    load_builtin_registrations()
    return tuple(sorted(TASKS))


def available_methods() -> tuple[str, ...]:
    load_builtin_registrations()
    return tuple(sorted(METHODS))
