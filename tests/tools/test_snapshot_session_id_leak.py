"""Cross-session HERMES_SESSION_ID leak via the shared bash snapshot.

Regression coverage for the bug where a single long-lived backend serves many
sessions through ONE ``_active_environments["default"]`` LocalEnvironment (the
messaging gateway, TUI, and desktop/web dashboard all collapse the terminal to
"default"). That environment persists a bash *session snapshot* file and
``source``s it before every command. ``export -p`` dumped the FIRST session's
``HERMES_SESSION_ID`` into the snapshot, so every LATER session ``source``d that
stale value and its ``echo $HERMES_SESSION_ID`` reported a FOREIGN session's id
— overriding the correct per-command Popen env injected by
``_inject_session_context_env``.

The fix strips command-scoped Hermes authority vars (session/UI/delivery,
Kanban ownership, and delegated-child lineage) from the snapshot at both dump
sites in ``tools/environments/base.py``; they are re-injected fresh on every
command.
"""

import os
import re
import sys

import pytest

from tools.environments.base import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


# ---------------------------------------------------------------------------
# Unit: the exclusion contract covers every Hermes-owned ephemeral class.
# ---------------------------------------------------------------------------

def test_regex_matches_bridged_session_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    # Every var the gateway bridges must be excluded.
    from gateway.session_context import _VAR_MAP

    for name in _VAR_MAP:
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"

    for name in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
        "SHLVL",
    ):
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"


def test_export_snippet_shape():
    snippet = _export_dump_excluding_session_vars("/tmp/snap.tmp.$BASHPID")
    assert "export -p" in snippet
    # Unset-by-name (not line-grep): multi-line declare values must not leave
    # continuation lines in the snapshot (issue #71296).
    assert "unset" in snippet
    assert '"${!HERMES_SESSION_@}"' in snippet
    assert '"${!HERMES_CRON_AUTO_DELIVER_@}"' in snippet
    assert '"${!HERMES_KANBAN_@}"' in snippet
    assert "HERMES_UI_SESSION_ID" in snippet
    assert "HERMES_DELEGATED_CHILD_CONTEXT" in snippet
    assert "SHLVL" in snippet
    assert "builtin export -n" in snippet
    assert snippet.index("builtin export -n") < snippet.index("builtin unset")
    assert "builtin export -p" in snippet
    assert '"$BASH" --noprofile --norc -p -c' in snippet
    assert "grep -vE" not in snippet
    assert "/tmp/snap.tmp.$BASHPID" in snippet
    # The redirection must be attached to a brace group wrapping the dump,
    # NOT to a pipeline segment: a redirect on a pipeline segment expands
    # $BASHPID inside that segment's subshell (a different PID than the parent
    # that expands the follow-up ``mv`` operand), silently orphaning the dump
    # and breaking snapshot env persistence entirely.
    assert snippet.lstrip().startswith("{ ")
    assert "|| true; }" in snippet
    assert snippet.rstrip().endswith("> /tmp/snap.tmp.$BASHPID")


# ---------------------------------------------------------------------------
# Integration: real LocalEnvironment, two sessions, no cross-contamination.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_shared_snapshot_no_cross_session_leak(tmp_path):
    import threading

    from gateway.session_context import _VAR_MAP, _UNSET, set_session_vars
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    try:
        def run_as(sid):
            out = {}

            def worker():
                for v in _VAR_MAP.values():
                    v.set(_UNSET)
                set_session_vars(session_key="k" + sid, session_id=sid, source="desktop")
                out["r"] = env.execute('echo "[$HERMES_SESSION_ID]"')

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            return out["r"].get("output", "")

        out_a = run_as("SIDAAA")
        out_b = run_as("SIDBBB")

        assert "SIDAAA" in out_a, f"session A saw {out_a!r}"
        # The core assertion: B must see its OWN id, not A's leaked via snapshot.
        assert "SIDBBB" in out_b, f"session B saw {out_b!r}"
        assert "SIDAAA" not in out_b, f"session B leaked A's id: {out_b!r}"

        # And the snapshot file must not carry the session id at all.
        snap = env._snapshot_path
        if os.path.exists(snap):
            with open(snap) as f:
                assert "HERMES_SESSION_ID" not in f.read()
    finally:
        env.cleanup()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_shared_snapshot_preserves_parent_child_kanban_boundary(monkeypatch, tmp_path):
    """A child cannot regain parent Kanban env or contaminate the parent."""
    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    command = (
        "printf '%s|%s|%s|%s\\n' "
        '"${HERMES_KANBAN_TASK-unset}" '
        '"${HERMES_DELEGATED_CHILD_CONTEXT-unset}" '
        '"${SNAPSHOT_CANARY-unset}" '
        '"${SHLVL-unset}"'
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        parent_before = env.execute(
            "export SNAPSHOT_CANARY=kept; " + command
        )["output"]
        readonly_result = env.execute(
            "IFS=_; export HERMES_KANBAN_TASK=t_readonly; "
            "readonly HERMES_KANBAN_TASK; "
            "export HERMES_DELEGATED_CHILD_CONTEXT=readonly-child; "
            "readonly HERMES_DELEGATED_CHILD_CONTEXT; "
            "builtin() { :; }; export -f builtin; "
            "BASH=/definitely/not/a/bash"
        )
        assert readonly_result["returncode"] == 0, readonly_result["output"]
        with delegated_child_context():
            child = env.execute(command)["output"]
        parent_after = env.execute(command)["output"]

        parent_before_state = re.search(
            r"t_parent\|unset\|kept\|(\d+)", parent_before
        )
        child_state = re.search(r"unset\|1\|kept\|(\d+)", child)
        parent_after_state = re.search(
            r"t_parent\|unset\|kept\|(\d+)", parent_after
        )
        assert parent_before_state, parent_before
        assert child_state, child
        assert parent_after_state, parent_after
        assert {
            parent_before_state.group(1),
            child_state.group(1),
            parent_after_state.group(1),
        } == {parent_before_state.group(1)}

        snapshot = open(env._snapshot_path, encoding="utf-8").read()
        assert 'declare -x SNAPSHOT_CANARY="kept"' in snapshot
        assert not re.search(r"^declare -x SHLVL=", snapshot, re.MULTILINE)
        assert "HERMES_KANBAN_" not in snapshot
        assert "HERMES_DELEGATED_CHILD_CONTEXT" not in snapshot
    finally:
        env.cleanup()
