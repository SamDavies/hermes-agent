"""Regression coverage for stale dispatcher-worker Kanban mutations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        yield conn


def _claim_worker(conn, monkeypatch):
    worker_id = kb.create_task(conn, title="worker", assignee="worker")
    candidate_id = kb.create_task(conn, title="candidate", assignee="worker")
    claimed = kb.claim_task(conn, worker_id, claimer="test-host:worker")
    assert claimed is not None
    current = kb.get_task(conn, worker_id)
    assert current is not None
    assert current.current_run_id is not None
    assert current.claim_lock
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(current.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", current.claim_lock)
    return worker_id, candidate_id, current.current_run_id


def test_reclaimed_worker_cannot_comment_or_archive_another_task(
    board, monkeypatch
):
    worker_id, candidate_id, _ = _claim_worker(board, monkeypatch)

    kb.add_comment(board, candidate_id, "worker", "live write")

    # Simulate an operator archiving/reclaiming the resolver while its process
    # is still alive. The stale child process retains the old env tuple.
    with monkeypatch.context() as operator:
        operator.delenv("HERMES_KANBAN_TASK")
        operator.delenv("HERMES_KANBAN_RUN_ID")
        operator.delenv("HERMES_KANBAN_CLAIM_LOCK")
        assert kb.archive_task(board, worker_id)

    with pytest.raises(PermissionError, match="no longer owns active run"):
        kb.add_comment(board, candidate_id, "worker", "orphan write")
    with pytest.raises(PermissionError, match="no longer owns active run"):
        kb.archive_task(board, candidate_id)

    comments = kb.list_comments(board, candidate_id)
    assert [comment.body for comment in comments] == ["live write"]
    assert kb.get_task(board, candidate_id).status == "ready"


def test_partial_dispatcher_authority_is_rejected(board, monkeypatch):
    worker_id = kb.create_task(board, title="worker", assignee="worker")
    candidate_id = kb.create_task(board, title="candidate", assignee="worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")

    with pytest.raises(PermissionError, match="missing its trusted"):
        kb.add_comment(board, candidate_id, "worker", "not authorized")


def test_legacy_task_only_scope_is_live_state_fenced(board, monkeypatch):
    worker_id = kb.create_task(board, title="worker", assignee="worker")
    candidate_id = kb.create_task(board, title="candidate", assignee="worker")
    claimed = kb.claim_task(board, worker_id, claimer="test-host:legacy")
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_id)

    kb.add_comment(board, candidate_id, "worker", "live legacy write")

    with monkeypatch.context() as operator:
        operator.delenv("HERMES_KANBAN_TASK")
        assert kb.archive_task(board, worker_id)

    with pytest.raises(PermissionError, match="no longer owns a live run"):
        kb.add_comment(board, candidate_id, "worker", "stale legacy write")


def test_legacy_live_worker_completion_finishes_housekeeping(board, monkeypatch):
    worker_id = kb.create_task(board, title="legacy worker", assignee="worker")
    claimed = kb.claim_task(board, worker_id, claimer="test-host:legacy")
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_id)

    assert kb.complete_task(board, worker_id, summary="legacy completion")
    assert kb.get_task(board, worker_id).status == "done"


def test_live_worker_completion_keeps_authorized_housekeeping(board, monkeypatch):
    worker_id, child_id, run_id = _claim_worker(board, monkeypatch)
    kb.link_tasks(board, worker_id, child_id)

    assert kb.complete_task(
        board,
        worker_id,
        summary="completed worker and released child",
        expected_run_id=run_id,
    )

    completed = kb.get_task(board, worker_id)
    child = kb.get_task(board, child_id)
    assert completed.status == "done"
    assert completed.current_run_id is None
    assert child.status == "ready"


def test_live_worker_own_archive_finishes_cleanly(board, monkeypatch):
    worker_id, _, _ = _claim_worker(board, monkeypatch)
    assert kb.archive_task(board, worker_id)
    assert kb.get_task(board, worker_id).status == "archived"


def test_worker_recovery_actions_are_operator_only(board, monkeypatch):
    worker_id, _, _ = _claim_worker(board, monkeypatch)
    signalled = []

    def _unexpected_termination(*args, **kwargs):
        signalled.append((args, kwargs))
        raise AssertionError("worker recovery must fail before process control")

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _unexpected_termination)
    with pytest.raises(PermissionError, match="operator recovery actions"):
        kb.reclaim_task(board, worker_id)
    with pytest.raises(PermissionError, match="operator recovery actions"):
        kb.reassign_task(board, worker_id, "replacement", reclaim_first=True)

    assert signalled == []
    current = kb.get_task(board, worker_id)
    assert current.status == "running"
    assert current.assignee == "worker"


def test_stale_worker_cannot_signal_reclaim_target(board, monkeypatch):
    worker_id, target_id, _ = _claim_worker(board, monkeypatch)
    with monkeypatch.context() as operator:
        operator.delenv("HERMES_KANBAN_TASK")
        operator.delenv("HERMES_KANBAN_RUN_ID")
        operator.delenv("HERMES_KANBAN_CLAIM_LOCK")
        assert kb.claim_task(board, target_id, claimer="test-host:target")
        assert kb.archive_task(board, worker_id)

    signalled = []

    def _unexpected_termination(*args, **kwargs):
        signalled.append((args, kwargs))
        raise AssertionError("termination must not run for a stale caller")

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _unexpected_termination)
    with pytest.raises(PermissionError, match="operator recovery actions"):
        kb.reclaim_task(board, target_id)
    assert signalled == []
    assert kb.get_task(board, target_id).status == "running"


def test_operator_reclaim_signals_before_waiting_for_write_lock(board, monkeypatch):
    target_id = kb.create_task(board, title="wedged worker", assignee="worker")
    assert kb.claim_task(board, target_id, claimer="test-host:target")
    kb._set_worker_pid(board, target_id, 12345)

    signalled = []

    def _termination(pid, claim_lock, *, signal_fn=None):
        signalled.append((pid, claim_lock))
        return {
            "prev_pid": pid,
            "host_local": True,
            "termination_attempted": True,
            "terminated": True,
            "sigkill": False,
        }

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _termination)
    blocker = kb.connect()
    try:
        blocker.execute("BEGIN IMMEDIATE")
        board.execute("PRAGMA busy_timeout=0")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            kb.reclaim_task(board, target_id)
    finally:
        blocker.rollback()
        blocker.close()

    assert len(signalled) == 1
    assert signalled[0][0] == 12345
    assert kb.get_task(board, target_id).status == "running"


def test_dispatcher_workers_cannot_manage_board_files(board, monkeypatch):
    worker_id, _, _ = _claim_worker(board, monkeypatch)
    with monkeypatch.context() as operator:
        operator.delenv("HERMES_KANBAN_TASK")
        operator.delenv("HERMES_KANBAN_RUN_ID")
        operator.delenv("HERMES_KANBAN_CLAIM_LOCK")
        kb.create_board("victim")

    for mutation in (
        lambda: kb.create_board("orphan"),
        lambda: kb.set_current_board("victim"),
        lambda: kb.write_board_metadata("victim", name="changed"),
        lambda: kb.remove_board("victim"),
    ):
        with pytest.raises(PermissionError, match="board management refused"):
            mutation()

    assert not kb.board_exists("orphan")
    assert kb.board_exists("victim")
    assert kb.get_current_board() == "default"
    assert kb.read_board_metadata("victim")["name"] == "Victim"


def test_operator_without_worker_scope_remains_unrestricted(board):
    task_id = kb.create_task(board, title="operator target", assignee="worker")
    kb.add_comment(board, task_id, "operator", "allowed")
    assert kb.archive_task(board, task_id)
    assert kb.get_task(board, task_id).status == "archived"
