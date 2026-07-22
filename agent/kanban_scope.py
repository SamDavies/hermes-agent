"""Agent-local Kanban lifecycle scope for delegated work."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, MutableMapping, Optional


_TASKLESS_KANBAN_SCOPE: ContextVar[bool] = ContextVar(
    "taskless_kanban_scope",
    default=False,
)

_KANBAN_PROJECT_CONTEXT_VARS = frozenset(
    {"HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_BRANCH"}
)


def is_taskless_kanban_scope() -> bool:
    """Return whether delegated work reports through a parent orchestrator."""
    marker = os.environ.get("HERMES_KANBAN_TASKLESS", "").strip().lower()
    return _TASKLESS_KANBAN_SCOPE.get() or marker in {"1", "true", "yes", "on"}


def is_taskless_kanban_agent(agent: object) -> bool:
    """Return whether an agent delegates board lifecycle to its parent."""
    return getattr(agent, "platform", None) == "subagent"


def taskless_kanban_subprocess_overrides() -> dict[str, Optional[str]]:
    """Return environment overrides for a delegated child subprocess."""
    return {
        "HERMES_KANBAN_TASK": None,
        "HERMES_KANBAN_RUN_ID": None,
        "HERMES_KANBAN_CLAIM_LOCK": None,
        "HERMES_KANBAN_TASKLESS": "1",
    }


def apply_taskless_kanban_subprocess_env(
    env: MutableMapping[str, str],
) -> MutableMapping[str, str]:
    """Apply the current delegated scope to a subprocess environment."""
    marker = str(env.get("HERMES_KANBAN_TASKLESS", "")).strip().lower()
    if (
        not is_taskless_kanban_scope()
        and marker not in {"1", "true", "yes", "on"}
    ):
        return env
    for name in list(env):
        if (
            name.startswith("HERMES_KANBAN_")
            and name not in _KANBAN_PROJECT_CONTEXT_VARS
            and name != "HERMES_KANBAN_TASKLESS"
        ):
            env.pop(name, None)
    env["HERMES_KANBAN_TASKLESS"] = "1"
    return env


@contextmanager
def taskless_kanban_scope(enabled: bool = True) -> Iterator[None]:
    """Bind delegated Kanban scope for one agent execution context."""
    token = _TASKLESS_KANBAN_SCOPE.set(bool(enabled))
    try:
        yield
    finally:
        _TASKLESS_KANBAN_SCOPE.reset(token)
