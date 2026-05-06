import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harbor.environments.base import ExecResult
from harbor.models.agent.name import AgentName
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial
from harbor.utils.logger import logger


def _write_snapshot_metadata(
    path: Path,
    *,
    snapshot_id: str,
    image_tag: str,
    archive_path: Path,
    round_num: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(f"archive-{round_num}")
    path.write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "image_tag": image_tag,
                "image_ref": f"sha256:{snapshot_id}",
                "archive_path": str(archive_path.resolve()),
                "round": round_num,
            },
            indent=2,
        )
    )


def _write_resume_source_config(source_dir: Path, *, agent_name: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": "/tmp/task"},
                "trial_name": "source-trial",
                "agent": {"name": agent_name},
            }
        )
    )


def test_build_verifier_failure_message_uses_single_verifier_dir_summary(tmp_path: Path):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths

    message = trial._build_verifier_failure_message(
        2,
        "Verifier exited with code 1 during round 2",
        include_exit_code=True,
    )

    assert "Verifier exited with code 1 during round 2" in message
    assert str(trial_paths.verifier_dir) in message
    assert "test-stdout.txt" in message
    assert "test-stderr.txt" in message
    assert "test-exit-code.txt" in message
    assert "verification-error.txt" in message
    assert str(trial_paths.test_stdout_path) not in message
    assert str(trial_paths.test_stderr_path) not in message


@pytest.mark.asyncio
async def test_capture_environment_state_snapshot_writes_round_and_latest_files(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    async def fake_capture(snapshot_id: str, archive_path: Path, restart_container: bool):
        assert restart_container is True
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text("round-archive")
        return {
            "snapshot_id": snapshot_id,
            "image_tag": "hbstate__demo-round-2",
            "image_ref": "sha256:demo-round-2",
            "archive_path": str(archive_path.resolve()),
        }

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._task = SimpleNamespace(is_multiround=True)
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(
        capture_state_snapshot=AsyncMock(side_effect=fake_capture)
    )
    trial._agent = SimpleNamespace(name=lambda: AgentName.ORACLE.value)
    trial._latest_snapshot_parent_id = "parent-snapshot"
    trial._result = SimpleNamespace(environment_state=None)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(
            multiround_state_cache_policy="success",
            multiround_resume_source="/tmp/source-trial",
        ),
    )

    await trial._capture_environment_state_snapshot(
        round_num=2,
        round_reward=1.0,
        round_status="completed",
    )

    round_snapshot_path = trial_paths.round_state_snapshot_path(2)
    latest_snapshot_path = trial_paths.state_snapshot_path
    latest_archive_path = trial_paths.state_image_archive_path

    assert round_snapshot_path.exists()
    assert latest_snapshot_path.exists()
    assert latest_archive_path.exists()

    round_payload = json.loads(round_snapshot_path.read_text())
    latest_payload = json.loads(latest_snapshot_path.read_text())

    assert round_payload["round"] == 2
    assert round_payload["parent_snapshot_id"] == "parent-snapshot"
    assert latest_payload["round"] == 2
    assert latest_payload["snapshot_id"] == "demo__abc123__round-2__state"
    assert latest_payload["archive_path"] == str(
        latest_archive_path.expanduser().absolute()
    )
    assert trial.result.environment_state.snapshot_id == "demo__abc123__round-2__state"
    assert trial.result.environment_state.parent_snapshot_id == "parent-snapshot"
    assert trial._latest_snapshot_parent_id == "demo__abc123__round-2__state"


@pytest.mark.asyncio
async def test_capture_environment_state_snapshot_also_writes_claude_session_snapshot(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    session_log = trial_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    session_log.parent.mkdir(parents=True, exist_ok=True)
    session_log.write_text("round-2-session")

    async def fake_capture(snapshot_id: str, archive_path: Path, restart_container: bool):
        assert restart_container is True
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text("round-archive")
        return {
            "snapshot_id": snapshot_id,
            "image_tag": "hbstate__demo-round-2",
            "image_ref": "sha256:demo-round-2",
            "archive_path": str(archive_path.resolve()),
        }

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._task = SimpleNamespace(is_multiround=True)
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(
        capture_state_snapshot=AsyncMock(side_effect=fake_capture),
        download_dir=AsyncMock(),
        exec=AsyncMock(),
        is_mounted=True,
    )
    trial._agent = SimpleNamespace(
        name=lambda: AgentName.CLAUDE_CODE.value,
        _round_starts=[(1, "2026-03-07T10:00:00+00:00"), (2, "2026-03-07T10:05:00+00:00")],
    )
    trial._latest_snapshot_parent_id = None
    trial._result = SimpleNamespace(environment_state=None)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(
            multiround_state_cache_policy="success",
            multiround_resume_source=None,
        ),
    )

    await trial._capture_environment_state_snapshot(
        round_num=2,
        round_reward=1.0,
        round_status="completed",
    )

    round_session_log = (
        trial_paths.agent_round_sessions_dir(2) / "projects" / "-app" / "session.jsonl"
    )
    assert round_session_log.exists()
    assert round_session_log.read_text() == "round-2-session"
    round_starts_path = (
        trial_paths.agent_round_sessions_dir(2) / Trial._CLAUDE_ROUND_STARTS_FILENAME
    )
    assert round_starts_path.exists()
    assert json.loads(round_starts_path.read_text()) == [
        {"round": 1, "started_at": "2026-03-07T10:00:00+00:00"},
        {"round": 2, "started_at": "2026-03-07T10:05:00+00:00"},
    ]


@pytest.mark.asyncio
async def test_capture_claude_session_snapshot_fixes_permissions_for_mounted_env(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    session_log = trial_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    session_log.parent.mkdir(parents=True, exist_ok=True)
    session_log.write_text("round-2-session")

    environment = SimpleNamespace(
        is_mounted=True,
        exec=AsyncMock(),
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = trial_paths
    trial._environment = environment
    trial._agent = SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(),
    )

    await trial._capture_claude_session_snapshot(round_num=2)

    copied_session_log = (
        trial_paths.agent_round_sessions_dir(2) / "projects" / "-app" / "session.jsonl"
    )
    assert copied_session_log.read_text() == "round-2-session"
    environment.exec.assert_awaited_once()
    permission_fix_command = environment.exec.await_args.args[0]
    assert "/logs/agent/sessions" in permission_fix_command
    assert "chown -R" in permission_fix_command
    assert "chmod -R u+rwX,go+rX" in permission_fix_command


@pytest.mark.asyncio
async def test_capture_claude_session_snapshot_raises_when_sessions_missing(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    environment = SimpleNamespace(
        is_mounted=True,
        exec=AsyncMock(),
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = trial_paths
    trial._environment = environment
    trial._agent = SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="Claude sessions directory was not found"):
        await trial._capture_claude_session_snapshot(round_num=2)

    assert not trial_paths.agent_round_sessions_dir(2).exists()
    environment.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_capture_environment_state_snapshot_also_writes_terminus_runtime_snapshot(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    async def fake_capture(snapshot_id: str, archive_path: Path, restart_container: bool):
        assert restart_container is True
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text("round-archive")
        return {
            "snapshot_id": snapshot_id,
            "image_tag": "hbstate__demo-round-2",
            "image_ref": "sha256:demo-round-2",
            "archive_path": str(archive_path.resolve()),
        }

    async def fake_capture_resume_state(snapshot_path: Path, round_num: int):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps({"round": round_num, "chat_messages": [{"role": "user"}]})
        )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._task = SimpleNamespace(is_multiround=True)
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(
        capture_state_snapshot=AsyncMock(side_effect=fake_capture),
        is_mounted=True,
    )
    trial._agent = SimpleNamespace(
        name=lambda: AgentName.TERMINUS_2.value,
        capture_resume_state=AsyncMock(side_effect=fake_capture_resume_state),
    )
    trial._latest_snapshot_parent_id = None
    trial._result = SimpleNamespace(environment_state=None)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(
            multiround_state_cache_policy="success",
            multiround_resume_source=None,
        ),
    )

    await trial._capture_environment_state_snapshot(
        round_num=2,
        round_reward=1.0,
        round_status="completed",
    )

    round_runtime_snapshot = trial_paths.terminus_round_runtime_state_path(2)
    assert round_runtime_snapshot.exists()
    assert json.loads(round_runtime_snapshot.read_text())["round"] == 2


@pytest.mark.asyncio
async def test_capture_environment_state_snapshot_captures_terminus_runtime_before_docker_snapshot(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    calls: list[str] = []

    async def fake_capture(snapshot_id: str, archive_path: Path, restart_container: bool):
        calls.append("capture_state_snapshot")
        assert restart_container is True
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text("round-archive")
        return {
            "snapshot_id": snapshot_id,
            "image_tag": "hbstate__demo-round-2",
            "image_ref": "sha256:demo-round-2",
            "archive_path": str(archive_path.resolve()),
        }

    async def fake_capture_resume_state(snapshot_path: Path, round_num: int):
        calls.append("capture_resume_state")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps({"round": round_num}))

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._task = SimpleNamespace(is_multiround=True)
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(
        capture_state_snapshot=AsyncMock(side_effect=fake_capture),
        is_mounted=True,
    )
    trial._agent = SimpleNamespace(
        name=lambda: AgentName.TERMINUS_2.value,
        capture_resume_state=AsyncMock(side_effect=fake_capture_resume_state),
    )
    trial._latest_snapshot_parent_id = None
    trial._result = SimpleNamespace(environment_state=None)
    trial.config = SimpleNamespace(
        trial_name="demo__abc123",
        verifier=SimpleNamespace(
            multiround_state_cache_policy="success",
            multiround_resume_source=None,
        ),
    )

    await trial._capture_environment_state_snapshot(
        round_num=2,
        round_reward=1.0,
        round_status="completed",
    )

    assert calls[:2] == ["capture_resume_state", "capture_state_snapshot"]


def test_resolve_resume_state_prefers_requested_round_snapshot(tmp_path: Path):
    source_dir = tmp_path / "resume-source"
    source_paths = TrialPaths(trial_dir=source_dir)
    source_paths.mkdir()

    _write_snapshot_metadata(
        source_paths.round_state_snapshot_path(2),
        snapshot_id="snap-round-2",
        image_tag="hbstate__round-2",
        archive_path=source_paths.round_state_image_archive_path(2),
        round_num=2,
    )
    _write_snapshot_metadata(
        source_paths.state_snapshot_path,
        snapshot_id="snap-latest",
        image_tag="hbstate__latest",
        archive_path=source_paths.state_image_archive_path,
        round_num=3,
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_resume_state_image=None,
            multiround_resume_state_archive=None,
            multiround_resume_state_snapshot_id=None,
            multiround_resume_source=str(source_dir),
            multiround_start_round=3,
        )
    )

    image_ref, archive_path, snapshot_id = trial._resolve_resume_state()

    assert image_ref == "hbstate__round-2"
    assert snapshot_id == "snap-round-2"
    assert archive_path == str(source_paths.round_state_image_archive_path(2).resolve())


def test_resolve_resume_state_rejects_mismatched_latest_fallback(tmp_path: Path):
    source_dir = tmp_path / "resume-source"
    source_paths = TrialPaths(trial_dir=source_dir)
    source_paths.mkdir()

    _write_snapshot_metadata(
        source_paths.state_snapshot_path,
        snapshot_id="snap-latest",
        image_tag="hbstate__latest",
        archive_path=source_paths.state_image_archive_path,
        round_num=3,
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_resume_state_image=None,
            multiround_resume_state_archive=None,
            multiround_resume_state_snapshot_id=None,
            multiround_resume_source=str(source_dir),
            multiround_start_round=2,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="No matching resume snapshot metadata was found",
    ):
        trial._resolve_resume_state()


@pytest.mark.asyncio
async def test_copy_resume_agent_state_prefers_requested_round_session_snapshot(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    round_session_log = (
        source_paths.agent_round_sessions_dir(2) / "projects" / "-app" / "session.jsonl"
    )
    round_session_log.parent.mkdir(parents=True, exist_ok=True)
    round_session_log.write_text("round-2-session")
    (
        source_paths.agent_round_sessions_dir(2) / Trial._CLAUDE_ROUND_STARTS_FILENAME
    ).write_text(
        json.dumps(
            [
                {"round": 1, "started_at": "2026-03-07T10:00:00+00:00"},
                {"round": 2, "started_at": "2026-03-07T10:05:00+00:00"},
            ],
            indent=2,
        )
    )

    latest_session_log = (
        source_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    )
    latest_session_log.parent.mkdir(parents=True, exist_ok=True)
    latest_session_log.write_text("latest-session")

    (source_paths.agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.2",
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-03-07T10:00:00+00:00",
                        "source": "user",
                        "message": "round 1 prompt",
                        "extra": {"round": 1},
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-03-07T10:05:00+00:00",
                        "source": "agent",
                        "message": "round 2 answer",
                        "extra": {"round": 2},
                    },
                    {
                        "step_id": 3,
                        "timestamp": "2026-03-07T10:10:00+00:00",
                        "source": "agent",
                        "message": "round 3 answer",
                        "extra": {"round": 3},
                    },
                ],
            },
            indent=2,
        )
    )

    target_stale_file = target_paths.agent_sessions_dir / "stale.txt"
    target_stale_file.parent.mkdir(parents=True, exist_ok=True)
    target_stale_file.write_text("stale")

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._environment = SimpleNamespace(is_mounted=True)
    trial._agent = SimpleNamespace(
        name=lambda: AgentName.CLAUDE_CODE.value,
        _round_starts=[],
    )
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    await trial._copy_resume_agent_state(source_dir, start_round=3)

    restored_log = (
        target_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    )
    assert restored_log.read_text() == "round-2-session"
    assert not target_stale_file.exists()
    assert trial._agent._round_starts == [
        (1, "2026-03-07T10:00:00+00:00"),
        (2, "2026-03-07T10:05:00+00:00"),
    ]
    pre_resume_trajectory = json.loads(
        target_paths.agent_pre_resume_trajectory_path(2).read_text()
    )
    assert [step["step_id"] for step in pre_resume_trajectory["steps"]] == [1, 2]
    assert [step["message"] for step in pre_resume_trajectory["steps"]] == [
        "round 1 prompt",
        "round 2 answer",
    ]
    assert (
        pre_resume_trajectory["extra"]["multiround_pre_resume_snapshot"][
            "resume_source_trial"
        ]
        == source_dir.name
    )
    pre_resume_metadata = json.loads(
        target_paths.agent_pre_resume_metadata_path(2).read_text()
    )
    assert pre_resume_metadata["resume_source_trial"] == source_dir.name
    assert pre_resume_metadata["resume_completed_round"] == 2
    assert pre_resume_metadata["resume_into_round"] == 3
    pre_resume_session_log = (
        target_paths.agent_pre_resume_session_files_dir(2)
        / "projects"
        / "-app"
        / "session.jsonl"
    )
    assert pre_resume_session_log.read_text() == "round-2-session"
    pre_resume_session_metadata = json.loads(
        target_paths.agent_pre_resume_session_metadata_path(2).read_text()
    )
    assert pre_resume_session_metadata["round"] == 2
    assert pre_resume_session_metadata["source_kind"] == "round_snapshot"
    assert pre_resume_session_metadata["source_trial"] == str(source_dir)


def test_merge_resume_trajectory_adds_resume_metadata(tmp_path: Path):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    (source_paths.agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "source-session",
                "agent": {"name": "terminus-2", "version": "test"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-03-07T10:00:00+00:00",
                        "source": "user",
                        "message": "round 1 prompt",
                        "extra": {"round": 1},
                    }
                ],
            },
            indent=2,
        )
    )
    (target_paths.agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "target-session",
                "agent": {"name": "terminus-2", "version": "test"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-03-07T10:00:00+00:00",
                        "source": "user",
                        "message": "round 1 prompt",
                        "extra": {},
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-03-07T10:05:00+00:00",
                        "source": "agent",
                        "message": "round 2 answer",
                        "extra": {"round": 2},
                    },
                ],
            },
            indent=2,
        )
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_resume_state_snapshot_id="snapshot-round-1",
            multiround_resume_state_image="hbstate__round-1",
        )
    )

    trial._merge_resume_trajectory(source_dir, from_round=2)

    merged = json.loads((target_paths.agent_dir / "trajectory.json").read_text())
    assert [step["message"] for step in merged["steps"]] == [
        "round 1 prompt",
        "round 2 answer",
    ]
    assert merged["extra"]["multiround_resume"]["resume_source_trial"] == source_dir.name
    assert merged["extra"]["multiround_resume"]["resume_completed_round"] == 1
    assert merged["extra"]["multiround_resume"]["resume_into_round"] == 2
    assert merged["notes"].startswith("Multiround resume:")


@pytest.mark.asyncio
async def test_copy_resume_agent_state_falls_back_to_latest_sessions_when_round_matches(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    latest_session_log = (
        source_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    )
    latest_session_log.parent.mkdir(parents=True, exist_ok=True)
    latest_session_log.write_text("latest-round-2-session")

    _write_snapshot_metadata(
        source_paths.state_snapshot_path,
        snapshot_id="snap-round-2",
        image_tag="hbstate__round-2",
        archive_path=source_paths.state_image_archive_path,
        round_num=2,
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._environment = SimpleNamespace(is_mounted=True)
    trial._agent = SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    await trial._copy_resume_agent_state(source_dir, start_round=3)

    restored_log = (
        target_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    )
    assert restored_log.read_text() == "latest-round-2-session"
    pre_resume_session_metadata = json.loads(
        target_paths.agent_pre_resume_session_metadata_path(2).read_text()
    )
    assert pre_resume_session_metadata["source_kind"] == "latest_sessions_fallback"


@pytest.mark.asyncio
async def test_copy_resume_agent_state_rejects_mismatched_latest_session_fallback(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    latest_session_log = (
        source_paths.agent_sessions_dir / "projects" / "-app" / "session.jsonl"
    )
    latest_session_log.parent.mkdir(parents=True, exist_ok=True)
    latest_session_log.write_text("latest-round-3-session")

    _write_snapshot_metadata(
        source_paths.state_snapshot_path,
        snapshot_id="snap-round-3",
        image_tag="hbstate__round-3",
        archive_path=source_paths.state_image_archive_path,
        round_num=3,
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._agent = SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    with pytest.raises(
        RuntimeError,
        match="No matching Claude session snapshot was found",
    ):
        await trial._copy_resume_agent_state(source_dir, start_round=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_agent_name", "agent_double"),
    [
        (
            AgentName.CLAUDE_CODE.value,
            SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value),
        ),
        (
            AgentName.TERMINUS_2.value,
            SimpleNamespace(
                name=lambda: AgentName.TERMINUS_2.value,
                restore_resume_state=MagicMock(),
            ),
        ),
    ],
)
async def test_copy_resume_agent_state_skips_agent_state_restore_for_oracle_source(
    tmp_path: Path,
    target_agent_name: str,
    agent_double: SimpleNamespace,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()
    _write_resume_source_config(source_dir, agent_name=AgentName.ORACLE.value)
    (source_paths.agent_dir / "round_1_oracle.txt").write_text("oracle")

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._environment = SimpleNamespace(is_mounted=True)
    trial._agent = agent_double
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    await trial._copy_resume_agent_state(source_dir, start_round=2)

    assert not target_paths.agent_sessions_dir.exists()
    if target_agent_name == AgentName.TERMINUS_2.value:
        agent_double.restore_resume_state.assert_not_called()


@pytest.mark.asyncio
async def test_copy_resume_agent_state_restores_terminus_runtime_snapshot(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    runtime_snapshot = source_paths.terminus_round_runtime_state_path(2)
    runtime_snapshot.parent.mkdir(parents=True, exist_ok=True)
    runtime_snapshot.write_text(json.dumps({"round": 2, "chat_messages": []}))

    restore_resume_state = MagicMock()
    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._agent = SimpleNamespace(
        name=lambda: AgentName.TERMINUS_2.value,
        restore_resume_state=restore_resume_state,
    )
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    await trial._copy_resume_agent_state(source_dir, start_round=3)

    restore_resume_state.assert_called_once_with(runtime_snapshot)


@pytest.mark.asyncio
async def test_copy_resume_agent_state_uploads_claude_sessions_for_nonmounted_env(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    round_session_log = (
        source_paths.agent_round_sessions_dir(2) / "projects" / "-app" / "session.jsonl"
    )
    round_session_log.parent.mkdir(parents=True, exist_ok=True)
    round_session_log.write_text("round-2-session")

    environment = SimpleNamespace(
        is_mounted=False,
        exec=AsyncMock(),
        upload_dir=AsyncMock(),
    )

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = target_paths
    trial._environment = environment
    trial._agent = SimpleNamespace(name=lambda: AgentName.CLAUDE_CODE.value)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace()
    )

    await trial._copy_resume_agent_state(source_dir, start_round=3)

    environment.upload_dir.assert_awaited_once_with(
        source_dir=target_paths.agent_sessions_dir,
        target_dir="/logs/agent/sessions",
    )


@pytest.mark.asyncio
async def test_execute_multiround_resume_from_snapshot_skips_fast_forward(tmp_path: Path):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    captured_preserve_status: list[bool] = []

    def fake_copy_resume_round_results(
        source_trial_dir: Path, up_to_round: int, *, preserve_status: bool = False
    ) -> list[dict]:
        captured_preserve_status.append(preserve_status)
        return []

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(restored_from_snapshot=True)
    trial._task = SimpleNamespace(
        num_rounds=3,
        round_instruction=lambda round_num: f"instruction-{round_num}",
        round_config=lambda round_num: {
            "change_type": "extension",
            "change_types": ["extension"],
        },
    )
    trial._agent = SimpleNamespace(run_round=AsyncMock())
    trial._agent_timeout_sec = 1
    trial._result = SimpleNamespace(agent_execution=None, agent_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=3,
            multiround_max_round=3,
            multiround_resume_source=str(tmp_path / "resume-source"),
            multiround_resume_trial_name="demo-trial",
            disable=False,
        )
    )
    trial._copy_resume_round_results = fake_copy_resume_round_results
    trial._fast_forward_rounds = AsyncMock()
    trial._copy_resume_agent_state = AsyncMock()
    trial._invoke_hooks = AsyncMock()
    trial._verify_round = AsyncMock(return_value=1.0)
    trial._backup_round_verifier_output = MagicMock()
    trial._capture_environment_state_snapshot = AsyncMock()
    trial._maybe_download_logs = AsyncMock()
    trial._maybe_populate_agent_context = MagicMock()
    trial._merge_resume_trajectory = MagicMock()
    trial._aggregate_multiround_results = MagicMock()

    await trial._execute_multiround()

    trial._fast_forward_rounds.assert_not_called()
    trial._copy_resume_agent_state.assert_awaited_once()
    assert captured_preserve_status == [True]


@pytest.mark.asyncio
async def test_execute_multiround_records_exception_info_for_agent_timeout(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    async def timeout_run_round(**kwargs):
        raise asyncio.TimeoutError()

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(restored_from_snapshot=False)
    trial._task = SimpleNamespace(
        num_rounds=3,
        round_instruction=lambda round_num: f"instruction-{round_num}",
        round_config=lambda round_num: {
            "change_type": "extension",
            "change_types": ["extension"],
        },
    )
    trial._agent = SimpleNamespace(run_round=timeout_run_round)
    trial._agent_timeout_sec = 7
    trial._result = SimpleNamespace(
        agent_execution=None,
        agent_result=None,
        verifier_result=None,
        exception_info=None,
    )
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=1,
            multiround_max_round=None,
            multiround_resume_source=None,
            multiround_resume_trial_name=None,
            disable=False,
        )
    )
    trial._invoke_hooks = AsyncMock()
    trial._backup_round_verifier_output = MagicMock()
    trial._capture_environment_state_snapshot = AsyncMock()
    trial._maybe_download_logs = AsyncMock()
    trial._maybe_populate_agent_context = MagicMock()

    await trial._execute_multiround()

    assert trial.result.exception_info is not None
    assert trial.result.exception_info.exception_type == "AgentTimeoutError"
    assert "7 seconds during round 1" in trial.result.exception_info.exception_message
    assert trial_paths.exception_message_path.exists()
    assert trial.result.verifier_result.rewards["round_1"] == 0
    assert trial.result.verifier_result.rewards["round_2"] == 0
    assert trial.result.verifier_result.rewards["round_3"] == 0


@pytest.mark.asyncio
async def test_execute_multiround_writes_exception_file_for_verification_error(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = trial_paths
    trial._environment = SimpleNamespace(restored_from_snapshot=False)
    trial._task = SimpleNamespace(
        num_rounds=2,
        round_instruction=lambda round_num: f"instruction-{round_num}",
        round_config=lambda round_num: {
            "change_type": "extension",
            "change_types": ["extension"],
        },
    )
    trial._agent = SimpleNamespace(run_round=AsyncMock(return_value=None))
    trial._agent_timeout_sec = 7
    trial._result = SimpleNamespace(
        agent_execution=None,
        agent_result=None,
        verifier_result=None,
        exception_info=None,
    )
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=1,
            multiround_max_round=None,
            multiround_resume_source=None,
            multiround_resume_trial_name=None,
            disable=False,
        )
    )
    trial._invoke_hooks = AsyncMock()
    trial._verify_round = AsyncMock(side_effect=FileNotFoundError("No reward file found"))
    trial._backup_round_verifier_output = MagicMock()
    trial._capture_environment_state_snapshot = AsyncMock()
    trial._maybe_download_logs = AsyncMock()
    trial._maybe_populate_agent_context = MagicMock()

    await trial._execute_multiround()

    assert trial.result.exception_info is None
    assert trial_paths.exception_message_path.exists()
    exception_text = trial_paths.exception_message_path.read_text()
    assert "FileNotFoundError: No reward file found" in exception_text
    assert trial.result.verifier_result.rewards["round_1"] == 0
    assert trial.result.verifier_result.rewards["round_2"] == 0


def test_aggregate_multiround_results_zero_fills_unexecuted_future_rounds(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=3)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=None,
            multiround_max_round=None,
            multiround_resume_trial_name=None,
        )
    )

    trial._aggregate_multiround_results(
        [
            {"round": 1, "reward": 1.0, "status": "completed"},
        ]
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards["round_1"] == 1.0
    assert trial.result.verifier_result.rewards["round_2"] == 0
    assert trial.result.verifier_result.rewards["round_3"] == 0
    assert trial.result.verifier_result.rewards["reward"] == pytest.approx(1 / 3)
    assert trial.result.verifier_result.aggregate_window_start == 1
    assert trial.result.verifier_result.aggregate_window_end == 3


def test_aggregate_multiround_results_preserves_historical_rewards_for_resume(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=3)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=2,
            multiround_max_round=None,
            multiround_resume_trial_name="demo-trial",
        )
    )

    trial._aggregate_multiround_results(
        [
            {"round": 1, "reward": 1.0, "status": "completed"},
            {"round": 2, "reward": 0.0, "status": "completed"},
        ]
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards["round_1"] == 1.0
    assert trial.result.verifier_result.rewards["round_2"] == 0.0
    assert trial.result.verifier_result.rewards["round_3"] == 0
    assert trial.result.verifier_result.rewards["reward"] == pytest.approx(1 / 3)
    assert trial.result.verifier_result.aggregate_window_start == 1
    assert trial.result.verifier_result.aggregate_window_end == 3


def test_aggregate_multiround_results_ignores_fast_forwarded_history_for_child_trial(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=3)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=3,
            multiround_max_round=3,
            multiround_resume_trial_name=None,
        )
    )

    trial._aggregate_multiround_results(
        [
            {"round": 1, "reward": 1.0, "status": "fast_forwarded"},
            {"round": 2, "reward": 1.0, "status": "fast_forwarded"},
            {"round": 3, "reward": 1.0, "status": "completed"},
        ]
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards["round_3"] == 1.0
    assert "round_1" not in trial.result.verifier_result.rewards
    assert "round_2" not in trial.result.verifier_result.rewards
    assert trial.result.verifier_result.rewards["reward"] == 1.0
    assert trial.result.verifier_result.aggregate_window_start == 3
    assert trial.result.verifier_result.aggregate_window_end == 3


def test_aggregate_multiround_results_honors_custom_aggregate_window(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=3)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=2,
            multiround_max_round=3,
            multiround_aggregate_start_round=2,
            multiround_aggregate_end_round=2,
            multiround_resume_trial_name="demo-trial",
        )
    )

    trial._aggregate_multiround_results(
        [
            {"round": 1, "reward": 0.0, "status": "completed"},
            {"round": 2, "reward": 1.0, "status": "completed"},
            {"round": 3, "reward": 1.0, "status": "completed"},
        ]
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards == {
        "round_2": 1.0,
        "reward": 1.0,
    }
    assert trial.result.verifier_result.aggregate_window_start == 2
    assert trial.result.verifier_result.aggregate_window_end == 2


def test_aggregate_multiround_results_requires_task_num_rounds(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=None)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=1,
            multiround_max_round=3,
            multiround_aggregate_start_round=None,
            multiround_aggregate_end_round=None,
            multiround_resume_trial_name=None,
        )
    )

    with pytest.raises(
        ValueError,
        match="Multi-round aggregation requires task.num_rounds to be available",
    ):
        trial._aggregate_multiround_results(
            [
                {"round": 1, "reward": 1.0, "status": "completed"},
            ]
        )


def test_requires_snapshot_capability_skips_fresh_multiround_when_cache_policy_off():
    trial = object.__new__(Trial)
    trial._task = SimpleNamespace(is_multiround=True)
    trial._resume_state_image_ref = None
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_state_cache_policy="off",
            multiround_resume_source=None,
        )
    )

    assert trial._requires_snapshot_capability() is False


def test_requires_snapshot_capability_requires_resume_state_when_resuming():
    trial = object.__new__(Trial)
    trial._task = SimpleNamespace(is_multiround=True)
    trial._resume_state_image_ref = "hbstate__round-2"
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_state_cache_policy="off",
            multiround_resume_source="/tmp/resume-source",
        )
    )

    assert trial._requires_snapshot_capability() is True


@pytest.mark.parametrize(
    ("policy", "round_reward", "round_status", "expected"),
    [
        ("success", 1.0, "completed", True),
        ("success", 0.0, "completed", False),
        ("success", None, "verification_disabled", False),
        ("success", None, "verification_error: boom", False),
        ("success", None, "agent_timeout", False),
        ("all", 1.0, "completed", True),
        ("all", 0.0, "completed", True),
        ("all", None, "verification_disabled", True),
        ("all", None, "verification_error: boom", True),
        ("all", None, "agent_timeout", False),
        ("off", 1.0, "completed", False),
        ("off", None, "verification_disabled", False),
    ],
)
def test_should_capture_state_snapshot_matrix(
    policy: str,
    round_reward: float | None,
    round_status: str,
    expected: bool,
):
    trial = object.__new__(Trial)
    trial._task = SimpleNamespace(is_multiround=True)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(multiround_state_cache_policy=policy)
    )

    assert (
        trial._should_capture_state_snapshot(
            round_reward=round_reward,
            round_status=round_status,
        )
        is expected
    )


def test_copy_resume_round_results_relabels_history_as_fast_forwarded_and_copies_files(
    tmp_path: Path,
):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    (source_paths.verifier_dir / "round_1_test-stdout.txt").write_text("stdout-1")
    (source_paths.verifier_dir / "round_1_test-stderr.txt").write_text("stderr-1")
    (source_paths.verifier_dir / "round_1_test-exit-code.txt").write_text("0")
    (source_paths.verifier_dir / "round_1_reward.txt").write_text("1")
    (source_paths.verifier_dir / "round_2_reward.json").write_text('{"reward": 0}')
    (source_paths.verifier_dir / "round_2_verification-error.txt").write_text(
        "No reward file found after round 2"
    )
    (source_paths.verifier_dir / "multiround_results.json").write_text(
        json.dumps(
            [
                {"round": 1, "reward": 1.0, "status": "completed"},
                {"round": 2, "reward": 0.0, "status": "completed"},
                {"round": 3, "reward": 1.0, "status": "completed"},
            ],
            indent=2,
        )
    )

    trial = object.__new__(Trial)
    trial._trial_paths = target_paths

    copied_results = trial._copy_resume_round_results(source_dir, up_to_round=3)

    assert copied_results == [
        {"round": 1, "reward": 1.0, "status": "fast_forwarded"},
        {"round": 2, "reward": 0.0, "status": "fast_forwarded"},
    ]
    assert (target_paths.verifier_dir / "round_1_test-stdout.txt").read_text() == "stdout-1"
    assert (target_paths.verifier_dir / "round_1_test-stderr.txt").read_text() == "stderr-1"
    assert (target_paths.verifier_dir / "round_1_test-exit-code.txt").read_text() == "0"
    assert (target_paths.verifier_dir / "round_1_reward.txt").read_text() == "1"
    assert (target_paths.verifier_dir / "round_2_reward.json").read_text() == '{"reward": 0}'
    assert (
        target_paths.verifier_dir / "round_2_verification-error.txt"
    ).read_text() == "No reward file found after round 2"
    assert not (target_paths.verifier_dir / "round_3_reward.txt").exists()


@pytest.mark.asyncio
async def test_verify_round_writes_diagnostics_when_verifier_exits_nonzero(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    round_tests_dir = tmp_path / "task" / "round_2" / "tests"
    round_test_path = round_tests_dir / "test.sh"
    round_tests_dir.mkdir(parents=True, exist_ok=True)
    round_test_path.write_text("#!/bin/sh\nexit 7\n")

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = TrialPaths(trial_dir=trial_dir)
    trial._trial_paths.mkdir()
    trial._verifier_timeout_sec = 30
    trial._task = SimpleNamespace(
        paths=SimpleNamespace(
            round_tests_dir=lambda round_num: round_tests_dir,
            round_test_path=lambda round_num: round_test_path,
        ),
        config=SimpleNamespace(verifier=SimpleNamespace(env=None)),
    )
    trial._environment = SimpleNamespace(
        is_mounted=True,
        upload_dir=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                ExecResult(return_code=0),
                ExecResult(return_code=0),
                ExecResult(return_code=0),
                ExecResult(return_code=7),
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="Verifier exited with code 7 during round 2"):
        await trial._verify_round(2)

    assert trial._trial_paths.test_stdout_path.exists()
    assert trial._trial_paths.test_stderr_path.exists()
    assert (trial._trial_paths.verifier_dir / "test-exit-code.txt").read_text() == "7"
    error_text = (trial._trial_paths.verifier_dir / "verification-error.txt").read_text()
    assert "Verifier exited with code 7 during round 2" in error_text
    assert str(trial._trial_paths.verifier_dir) in error_text
    assert "test-stderr.txt" in error_text


@pytest.mark.asyncio
async def test_verify_round_writes_diagnostics_when_reward_file_missing(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    round_tests_dir = tmp_path / "task" / "round_2" / "tests"
    round_test_path = round_tests_dir / "test.sh"
    round_tests_dir.mkdir(parents=True, exist_ok=True)
    round_test_path.write_text("#!/bin/sh\nexit 0\n")

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_multiround_state")
    trial._trial_paths = TrialPaths(trial_dir=trial_dir)
    trial._trial_paths.mkdir()
    trial._verifier_timeout_sec = 30
    trial._task = SimpleNamespace(
        paths=SimpleNamespace(
            round_tests_dir=lambda round_num: round_tests_dir,
            round_test_path=lambda round_num: round_test_path,
        ),
        config=SimpleNamespace(verifier=SimpleNamespace(env=None)),
    )
    trial._environment = SimpleNamespace(
        is_mounted=True,
        upload_dir=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                ExecResult(return_code=0),
                ExecResult(return_code=0),
                ExecResult(return_code=0),
                ExecResult(return_code=0),
            ]
        ),
    )

    with pytest.raises(FileNotFoundError, match="No reward file found after round 2"):
        await trial._verify_round(2)

    assert (trial._trial_paths.verifier_dir / "test-exit-code.txt").read_text() == "0"
    error_text = (trial._trial_paths.verifier_dir / "verification-error.txt").read_text()
    assert "No reward file found after round 2" in error_text
    assert str(trial._trial_paths.verifier_dir) in error_text
    assert "test-stdout.txt" in error_text


def test_copy_resume_round_results_preserves_status_for_inplace_resume(tmp_path: Path):
    source_dir = tmp_path / "resume-source"
    target_dir = tmp_path / "resume-target"
    source_paths = TrialPaths(trial_dir=source_dir)
    target_paths = TrialPaths(trial_dir=target_dir)
    source_paths.mkdir()
    target_paths.mkdir()

    (source_paths.verifier_dir / "multiround_results.json").write_text(
        json.dumps(
            [
                {"round": 1, "reward": 1.0, "status": "completed"},
                {"round": 2, "reward": 0.0, "status": "verification_disabled"},
            ],
            indent=2,
        )
    )

    trial = object.__new__(Trial)
    trial._trial_paths = target_paths

    copied_results = trial._copy_resume_round_results(
        source_dir,
        up_to_round=2,
        preserve_status=True,
    )

    assert copied_results == [
        {"round": 1, "reward": 1.0, "status": "completed"},
    ]


def test_aggregate_multiround_results_treats_verification_disabled_round_as_zero(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    trial = object.__new__(Trial)
    trial._trial_paths = trial_paths
    trial._task = SimpleNamespace(num_rounds=3)
    trial._result = SimpleNamespace(verifier_result=None)
    trial.config = SimpleNamespace(
        verifier=SimpleNamespace(
            multiround_start_round=2,
            multiround_max_round=3,
            multiround_resume_trial_name=None,
        )
    )

    trial._aggregate_multiround_results(
        [
            {
                "round": 2,
                "reward": None,
                "status": "verification_disabled",
            },
            {
                "round": 3,
                "reward": None,
                "status": "verification_disabled",
            },
        ]
    )

    assert trial.result.verifier_result is not None
    assert trial.result.verifier_result.rewards["round_2"] == 0
    assert trial.result.verifier_result.rewards["round_3"] == 0
    assert trial.result.verifier_result.rewards["reward"] == 0
