"""Agent-local ownership tests for dispatcher Kanban lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def isolated_hermes_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run_agent, "_hermes_home", home)


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent_with_tools(*names: str, platform: str | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )


def test_terminal_capabilities_bind_parent_lifecycle(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    parent = _agent_with_tools(
        "kanban_show",
        "kanban_complete",
        "kanban_block",
        "kanban_heartbeat",
        platform="cli",
    )

    assert parent._kanban_task_id == "t_parent"
    assert parent._kanban_worker_guidance


def test_task_env_without_terminal_capabilities_is_taskless(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    child = _agent_with_tools("terminal", "read_file", platform="subagent")

    assert child._kanban_task_id is None
    assert child._kanban_worker_guidance == ""


def test_subagent_identity_is_taskless_even_with_malformed_tool_snapshot(monkeypatch):
    """Delegated identity remains authoritative if a tool snapshot regresses."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    child = _agent_with_tools(
        "terminal",
        "kanban_complete",
        "kanban_block",
        platform="subagent",
    )

    assert "kanban" in child.disabled_toolsets
    assert child._kanban_task_id is None
    assert child._kanban_worker_guidance == ""


@pytest.mark.parametrize("delegate_role", ["leaf", "orchestrator"])
def test_subagent_schema_and_refresh_keep_parent_lifecycle(
    monkeypatch, delegate_role
):
    """Both delegated roles retain implementation tools across MCP refreshes."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    with (
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        child = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=["terminal", "kanban"],
            disabled_toolsets=[],
            quiet_mode=True,
            platform="subagent",
            skip_context_files=True,
            skip_memory=True,
        )
    child._delegate_role = delegate_role

    assert "kanban" in child.disabled_toolsets
    assert not any(name.startswith("kanban_") for name in child.valid_tool_names)

    from tools.mcp_tool import refresh_agent_mcp_tools
    refresh_agent_mcp_tools(child)

    assert not any(name.startswith("kanban_") for name in child.valid_tool_names)


def test_activity_heartbeat_uses_agent_local_lifecycle_owner(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    parent = _agent_with_tools(
        "kanban_complete", "kanban_block", platform="cli"
    )
    child = _agent_with_tools("terminal", platform="subagent")

    heartbeat = MagicMock()
    with patch(
        "tools.kanban_tools.heartbeat_current_worker_from_env", heartbeat
    ):
        child._touch_activity("child evidence returned")
        heartbeat.assert_not_called()

        parent._touch_activity("parent validating evidence")
        heartbeat.assert_called_once_with()


def test_taskless_subagent_stop_does_not_enter_kanban_stop_loop(monkeypatch):
    """A parent's process env does not turn a child reply into a handoff."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    child = _agent_with_tools("web_search", platform="subagent")
    child.client = MagicMock()
    child._cached_system_prompt = "You are an evidence probe."
    child._use_prompt_caching = False
    child.tool_delay = 0
    child.compression_enabled = False
    child.save_trajectories = False
    child.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="CI evidence summary",
                    tool_calls=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )

    with (
        patch(
            "agent.kanban_stop.build_kanban_stop_nudge",
            return_value=None,
        ) as stop_nudge,
        patch.object(child, "_persist_session"),
        patch.object(child, "_save_trajectory"),
        patch.object(child, "_cleanup_task_resources"),
    ):
        result = child.run_conversation("inspect CI")

    stop_nudge.assert_not_called()
    assert result["api_calls"] == 1
    assert result["final_response"] == "CI evidence summary"


@pytest.mark.parametrize(
    "env_builder",
    ["foreground_terminal", "background_terminal", "model_subprocess"],
)
def test_taskless_scope_strips_board_identity_from_subprocesses(
    monkeypatch, env_builder
):
    board_env = {
        "HERMES_KANBAN_TASK": "t_parent",
        "HERMES_KANBAN_RUN_ID": "17",
        "HERMES_KANBAN_CLAIM_LOCK": "parent-lock",
        "HERMES_KANBAN_DB": "/board/kanban.db",
        "HERMES_KANBAN_BOARD": "main",
        "HERMES_KANBAN_ATTACHMENTS_ROOT": "/board/attachments",
        "HERMES_KANBAN_GOAL_MODE": "1",
        "HERMES_KANBAN_STOP_NUDGE": "1",
        "HERMES_KANBAN_WORKSPACE": "/workspace/project",
        "HERMES_KANBAN_BRANCH": "codex/fix",
    }
    for name, value in board_env.items():
        monkeypatch.setenv(name, value)

    from agent.kanban_scope import taskless_kanban_scope
    from tools.environments.local import (
        _make_run_env,
        _sanitize_subprocess_env,
        hermes_subprocess_env,
    )

    with taskless_kanban_scope():
        if env_builder == "foreground_terminal":
            child_env = _make_run_env({})
        elif env_builder == "background_terminal":
            child_env = _sanitize_subprocess_env(
                dict(board_env),
                {
                    "HERMES_KANBAN_TASK": "t_readded",
                    "HERMES_KANBAN_DB": "/readded/kanban.db",
                },
            )
        else:
            child_env = hermes_subprocess_env(inherit_credentials=True)

    assert child_env["HERMES_KANBAN_TASKLESS"] == "1"
    assert child_env["HERMES_KANBAN_WORKSPACE"] == "/workspace/project"
    assert child_env["HERMES_KANBAN_BRANCH"] == "codex/fix"
    assert not {
        name
        for name in child_env
        if name.startswith("HERMES_KANBAN_")
        and name
        not in {
            "HERMES_KANBAN_TASKLESS",
            "HERMES_KANBAN_WORKSPACE",
            "HERMES_KANBAN_BRANCH",
        }
    }


def test_taskless_cli_returns_lifecycle_to_parent(monkeypatch, capsys):
    from agent.kanban_scope import taskless_kanban_scope
    from hermes_cli.kanban import kanban_command

    with taskless_kanban_scope():
        result = kanban_command(SimpleNamespace(kanban_action="complete"))

    assert result == 2
    assert "parent orchestrator" in capsys.readouterr().err
