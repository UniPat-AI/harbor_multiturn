import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.cli.jobs import (
    _cleanup_trial_for_resume,
    _compute_task_definition_checksum,
    _resolve_resume_trial_dir,
    _would_enable_roundwise_multiround_attempt_selection,
    start,
    _validate_resume_agent_transition,
    _validate_resume_shape,
)
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths


def _write_source_trial_config(
    trial_dir: Path, source_agent_name: str, task_path: Path
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trial_name="source-trial",
        agent=AgentConfig(name=source_agent_name),
    )
    (trial_dir / "config.json").write_text(config.model_dump_json(indent=2))


def _write_source_trial_result(trial_dir: Path, *, task_checksum: str) -> None:
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_checksum": task_checksum,
            },
            indent=2,
        )
    )


def _write_multiround_source_trial_result(
    trial_dir: Path, *, completed_round: int, reward: float
) -> None:
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {
                    "rewards": {
                        f"round_{completed_round}": reward,
                    }
                },
                "exception_info": None,
            },
            indent=2,
        )
    )


def _write_multiround_source_trial_result_rewards(
    trial_dir: Path, *, rewards: dict[str, float]
) -> None:
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {
                    "rewards": rewards,
                },
                "exception_info": None,
            },
            indent=2,
        )
    )


def _write_round_snapshot_metadata(trial_dir: Path, *, round_num: int) -> None:
    trial_paths = TrialPaths(trial_dir=trial_dir)
    snapshot_path = trial_paths.round_state_snapshot_path(round_num)
    archive_path = trial_paths.round_state_image_archive_path(round_num)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(f"archive-{round_num}")
    snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_id": f"snap-{round_num}",
                "image_tag": f"hbstate__round-{round_num}",
                "archive_path": str(archive_path.resolve()),
                "round": round_num,
            },
            indent=2,
        )
    )


def _write_latest_snapshot_metadata(trial_dir: Path, *, round_num: int) -> None:
    trial_paths = TrialPaths(trial_dir=trial_dir)
    snapshot_path = trial_paths.state_snapshot_path
    archive_path = trial_paths.state_image_archive_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(f"latest-archive-{round_num}")
    snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_id": f"latest-snap-{round_num}",
                "image_tag": f"hbstate__latest-round-{round_num}",
                "archive_path": str(archive_path.resolve()),
                "round": round_num,
            },
            indent=2,
        )
    )


def _write_claude_round_session_snapshot(trial_dir: Path, *, round_num: int) -> None:
    trial_paths = TrialPaths(trial_dir=trial_dir)
    session_log = (
        trial_paths.agent_round_sessions_dir(round_num)
        / "projects"
        / "-app"
        / "session.jsonl"
    )
    session_log.parent.mkdir(parents=True, exist_ok=True)
    session_log.write_text(f"round-{round_num}-session")


def _write_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("task")
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")


def _write_multiround_task(task_dir: Path, *, num_rounds: int = 3) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("task")
    rounds = []
    for round_num in range(1, num_rounds + 1):
        round_dir = task_dir / f"round_{round_num}"
        (round_dir / "solution").mkdir(parents=True, exist_ok=True)
        (round_dir / "tests").mkdir(parents=True, exist_ok=True)
        (round_dir / "instruction.md").write_text(f"round {round_num}")
        (round_dir / "solution" / "solve.sh").write_text("#!/bin/bash\n")
        (round_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
        rounds.append(
            f'[[metadata.multiround.rounds]]\nround = {round_num}\nchange_types = ["extension"]\n'
        )
    task_dir.joinpath("task.toml").write_text(
        'version = "1.0"\n\n'
        f"[metadata.multiround]\nnum_rounds = {num_rounds}\n\n"
        + "\n".join(rounds)
    )


def test_resume_task_definition_checksum_ignores_terminal_marker_files(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=2)

    baseline = _compute_task_definition_checksum(task_dir)

    (task_dir / "passed.txt").write_text("")
    (task_dir / "conditionally_passed.txt").write_text("")
    (task_dir / "failed.txt").write_text("reason\n")
    (task_dir / "round_1" / "passed.txt").write_text("")
    (task_dir / "round_1" / "conditionally_passed.txt").write_text("")
    (task_dir / "round_2" / "failed.txt").write_text("reason\n")

    assert _compute_task_definition_checksum(task_dir) == baseline


def test_validate_resume_agent_transition_allows_oracle_to_oracle(tmp_path: Path):
    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=tmp_path / "task",
    )

    _validate_resume_agent_transition(
        AgentConfig(name="oracle"),
        resume_trial_dir,
    )


def test_validate_resume_agent_transition_allows_oracle_to_agent(tmp_path: Path):
    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=tmp_path / "task",
    )

    _validate_resume_agent_transition(
        AgentConfig(name="claude-code"),
        resume_trial_dir,
    )


def test_validate_resume_agent_transition_rejects_agent_to_oracle(tmp_path: Path):
    resume_trial_dir = tmp_path / "agent-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="claude-code",
        task_path=tmp_path / "task",
    )

    with pytest.raises(
        ValueError,
        match="does not support resuming into oracle",
    ):
        _validate_resume_agent_transition(
            AgentConfig(name="oracle"),
            resume_trial_dir,
        )


def test_validate_resume_agent_transition_rejects_nonoracle_cross_agent_resume(
    tmp_path: Path,
):
    resume_trial_dir = tmp_path / "agent-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="claude-code",
        task_path=tmp_path / "task",
    )

    with pytest.raises(
        ValueError,
        match="only supports same-agent continuation, plus oracle -> target-agent handoff",
    ):
        _validate_resume_agent_transition(
            AgentConfig(name="terminus-2"),
            resume_trial_dir,
        )


def test_cleanup_trial_for_resume_keeps_earlier_round_snapshots(tmp_path: Path):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    for round_num in (1, 2, 3):
        round_snapshot_path = trial_paths.round_state_snapshot_path(round_num)
        round_archive_path = trial_paths.round_state_image_archive_path(round_num)
        round_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        round_archive_path.write_text(f"archive-{round_num}")
        round_snapshot_path.write_text(
            json.dumps(
                {
                    "snapshot_id": f"snap-{round_num}",
                    "image_tag": f"hbstate__round-{round_num}",
                    "archive_path": str(round_archive_path.resolve()),
                    "round": round_num,
                },
                indent=2,
            )
        )

    trial_paths.state_snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-3",
                "image_tag": "hbstate__round-3",
                "archive_path": str(trial_paths.state_image_archive_path.absolute()),
                "round": 3,
            },
            indent=2,
        )
    )
    trial_paths.state_image_archive_path.write_text("latest-archive")
    trial_paths.agent_sessions_dir.mkdir(parents=True, exist_ok=True)
    (trial_paths.agent_sessions_dir / "latest.jsonl").write_text("live-session")

    for round_num in (1, 2, 3):
        round_session_log = (
            trial_paths.agent_round_sessions_dir(round_num)
            / "projects"
            / "-app"
            / "session.jsonl"
        )
        round_session_log.parent.mkdir(parents=True, exist_ok=True)
        round_session_log.write_text(f"session-{round_num}")
        terminus_state = trial_paths.terminus_round_runtime_state_path(round_num)
        terminus_state.parent.mkdir(parents=True, exist_ok=True)
        terminus_state.write_text(json.dumps({"round": round_num}))

    _cleanup_trial_for_resume(trial_dir, from_round=3)

    assert trial_paths.round_state_snapshot_path(1).exists()
    assert trial_paths.round_state_snapshot_path(2).exists()
    assert not trial_paths.round_state_snapshot_path(3).exists()
    assert trial_paths.agent_round_sessions_dir(1).exists()
    assert trial_paths.agent_round_sessions_dir(2).exists()
    assert not trial_paths.agent_round_sessions_dir(3).exists()
    assert trial_paths.terminus_round_runtime_state_path(1).exists()
    assert trial_paths.terminus_round_runtime_state_path(2).exists()
    assert not trial_paths.terminus_round_runtime_state_path(3).exists()
    assert not trial_paths.agent_sessions_dir.exists()

    latest_payload = json.loads(trial_paths.state_snapshot_path.read_text())
    assert latest_payload["round"] == 2
    assert latest_payload["snapshot_id"] == "snap-2"
    assert trial_paths.state_image_archive_path.exists()


def test_cleanup_trial_for_resume_preserves_earlier_oracle_logs_and_clears_exception(
    tmp_path: Path,
):
    trial_dir = tmp_path / "trial"
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    (trial_paths.agent_dir / "round_1_oracle.txt").write_text("round-1")
    (trial_paths.agent_dir / "round_2_oracle.txt").write_text("round-2")
    (trial_paths.agent_dir / "round_3_oracle.txt").write_text("round-3")
    (trial_paths.agent_dir / "round_3_exit-code.txt").write_text("17")
    trial_paths.test_stderr_path.write_text("old verifier stderr")
    trial_paths.exception_message_path.write_text("old exception")

    _cleanup_trial_for_resume(trial_dir, from_round=3)

    assert (trial_paths.agent_dir / "round_1_oracle.txt").exists()
    assert (trial_paths.agent_dir / "round_2_oracle.txt").exists()
    assert not (trial_paths.agent_dir / "round_3_oracle.txt").exists()
    assert not (trial_paths.agent_dir / "round_3_exit-code.txt").exists()
    assert not trial_paths.test_stderr_path.exists()
    assert not trial_paths.exception_message_path.exists()


def test_validate_resume_shape_requires_single_agent_and_single_attempt():
    config = JobConfig()
    config.agents = [AgentConfig(name="oracle"), AgentConfig(name="claude-code")]
    with pytest.raises(ValueError, match="exactly one agent"):
        _validate_resume_shape(config)
    _validate_resume_shape(JobConfig(n_attempts=2, agents=[AgentConfig(name="oracle")]))


def test_resolve_resume_trial_dir_selects_first_eligible_child_from_job_dir(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    job_dir = tmp_path / "2026-04-07__12-00-00"
    job_dir.mkdir()

    failed_trial = job_dir / "a-failed"
    TrialPaths(trial_dir=failed_trial).mkdir()
    _write_source_trial_config(failed_trial, source_agent_name="oracle", task_path=task_dir)
    _write_multiround_source_trial_result(failed_trial, completed_round=2, reward=0.0)
    _write_round_snapshot_metadata(failed_trial, round_num=2)

    missing_snapshot_trial = job_dir / "b-missing-snapshot"
    TrialPaths(trial_dir=missing_snapshot_trial).mkdir()
    _write_source_trial_config(
        missing_snapshot_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        missing_snapshot_trial,
        completed_round=2,
        reward=1.0,
    )

    selected_trial = job_dir / "c-selected"
    TrialPaths(trial_dir=selected_trial).mkdir()
    _write_source_trial_config(
        selected_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        selected_trial,
        completed_round=2,
        reward=1.0,
    )
    _write_round_snapshot_metadata(selected_trial, round_num=2)

    later_eligible_trial = job_dir / "d-later"
    TrialPaths(trial_dir=later_eligible_trial).mkdir()
    _write_source_trial_config(
        later_eligible_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        later_eligible_trial,
        completed_round=2,
        reward=1.0,
    )
    _write_round_snapshot_metadata(later_eligible_trial, round_num=2)

    resolved = _resolve_resume_trial_dir(job_dir, start_round=3)

    assert resolved == selected_trial.resolve()


def test_resolve_resume_trial_dir_skips_children_from_later_rounds(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=4)

    job_dir = tmp_path / "2026-04-07__12-00-00"
    job_dir.mkdir()

    later_round_trial = job_dir / "a-round-3"
    TrialPaths(trial_dir=later_round_trial).mkdir()
    _write_source_trial_config(
        later_round_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result_rewards(
        later_round_trial,
        rewards={"round_2": 1.0, "round_3": 1.0, "reward": 1.0},
    )
    _write_round_snapshot_metadata(later_round_trial, round_num=2)
    _write_latest_snapshot_metadata(later_round_trial, round_num=3)

    selected_trial = job_dir / "b-round-2"
    TrialPaths(trial_dir=selected_trial).mkdir()
    _write_source_trial_config(
        selected_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        selected_trial,
        completed_round=2,
        reward=1.0,
    )
    _write_round_snapshot_metadata(selected_trial, round_num=2)
    _write_latest_snapshot_metadata(selected_trial, round_num=2)

    resolved = _resolve_resume_trial_dir(job_dir, start_round=3)

    assert resolved == selected_trial.resolve()


def test_resolve_resume_trial_dir_preserves_explicit_trial_dir(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    trial_dir = tmp_path / "explicit-trial"
    TrialPaths(trial_dir=trial_dir).mkdir()
    _write_source_trial_config(trial_dir, source_agent_name="oracle", task_path=task_dir)
    _write_multiround_source_trial_result(trial_dir, completed_round=2, reward=1.0)
    _write_round_snapshot_metadata(trial_dir, round_num=2)

    resolved = _resolve_resume_trial_dir(trial_dir, start_round=3)

    assert resolved == trial_dir.resolve()


def test_start_accepts_resume_job_dir_and_uses_selected_child(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    job_dir = tmp_path / "2026-04-07__12-00-00"
    job_dir.mkdir()

    selected_trial = job_dir / "b-selected"
    TrialPaths(trial_dir=selected_trial).mkdir()
    _write_source_trial_config(
        selected_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        selected_trial,
        completed_round=2,
        reward=1.0,
    )
    _write_round_snapshot_metadata(selected_trial, round_num=2)

    later_trial = job_dir / "c-later"
    TrialPaths(trial_dir=later_trial).mkdir()
    _write_source_trial_config(
        later_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result_rewards(
        later_trial,
        rewards={"round_2": 1.0, "round_3": 1.0, "reward": 1.0},
    )
    _write_round_snapshot_metadata(later_trial, round_num=2)
    _write_latest_snapshot_metadata(later_trial, round_num=3)

    import harbor.job as job_module
    import harbor.cli.jobs as jobs_module

    captured = {}

    async def fake_run(self):
        captured["resume_source"] = self.config.verifier.multiround_resume_source
        captured["start_round"] = self.config.verifier.multiround_start_round
        return SimpleNamespace(stats=SimpleNamespace(evals={}))

    original_job_run = job_module.Job.run
    original_show_hint = jobs_module.show_registry_hint_if_first_run
    original_print_tables = jobs_module.print_job_results_tables

    job_module.Job.run = fake_run
    jobs_module.show_registry_hint_if_first_run = lambda console: None
    jobs_module.print_job_results_tables = lambda _: None
    try:
        start(
            path=task_dir,
            agent_name="oracle",
            resume_trial=job_dir,
            resume_round=3,
            output_jobs_dir=tmp_path / "output-jobs",
        )
    finally:
        job_module.Job.run = original_job_run
        jobs_module.show_registry_hint_if_first_run = original_show_hint
        jobs_module.print_job_results_tables = original_print_tables

    assert captured["resume_source"] == str(selected_trial.resolve())
    assert captured["start_round"] == 3


def test_start_in_place_resume_removes_stale_job_lock(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    job_dir = tmp_path / "2026-04-07__12-00-00"
    job_dir.mkdir()
    (job_dir / "lock.json").write_text('{"schema_version": 1}\n')

    selected_trial = job_dir / "selected-trial"
    TrialPaths(trial_dir=selected_trial).mkdir()
    _write_source_trial_config(
        selected_trial,
        source_agent_name="oracle",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        selected_trial,
        completed_round=1,
        reward=1.0,
    )
    _write_round_snapshot_metadata(selected_trial, round_num=1)

    import harbor.cli.jobs as jobs_module
    import harbor.job as job_module

    captured = {}

    async def fake_run(self):
        captured["lock_exists_at_run"] = (job_dir / "lock.json").exists()
        return SimpleNamespace(stats=SimpleNamespace(evals={}))

    original_job_run = job_module.Job.run
    original_show_hint = jobs_module.show_registry_hint_if_first_run
    original_print_tables = jobs_module.print_job_results_tables

    job_module.Job.run = fake_run
    jobs_module.show_registry_hint_if_first_run = lambda console: None
    jobs_module.print_job_results_tables = lambda _: None
    try:
        start(
            path=task_dir,
            agent_name="oracle",
            resume_trial=selected_trial,
            resume_round=2,
        )
    finally:
        job_module.Job.run = original_job_run
        jobs_module.show_registry_hint_if_first_run = original_show_hint
        jobs_module.print_job_results_tables = original_print_tables

    assert captured["lock_exists_at_run"] is False


def test_start_accepts_resume_job_dir_and_uses_selected_child_for_claude(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=4)

    job_dir = tmp_path / "2026-04-07__12-00-00"
    job_dir.mkdir()

    later_trial = job_dir / "a-later"
    TrialPaths(trial_dir=later_trial).mkdir()
    _write_source_trial_config(
        later_trial,
        source_agent_name="claude-code",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result_rewards(
        later_trial,
        rewards={"round_2": 1.0, "round_3": 1.0, "reward": 1.0},
    )
    _write_round_snapshot_metadata(later_trial, round_num=2)
    _write_latest_snapshot_metadata(later_trial, round_num=3)
    _write_claude_round_session_snapshot(later_trial, round_num=2)

    selected_trial = job_dir / "b-selected"
    TrialPaths(trial_dir=selected_trial).mkdir()
    _write_source_trial_config(
        selected_trial,
        source_agent_name="claude-code",
        task_path=task_dir,
    )
    _write_multiround_source_trial_result(
        selected_trial,
        completed_round=2,
        reward=1.0,
    )
    _write_round_snapshot_metadata(selected_trial, round_num=2)
    _write_latest_snapshot_metadata(selected_trial, round_num=2)
    _write_claude_round_session_snapshot(selected_trial, round_num=2)

    import harbor.job as job_module
    import harbor.cli.jobs as jobs_module

    captured = {}

    async def fake_run(self):
        captured["resume_source"] = self.config.verifier.multiround_resume_source
        captured["start_round"] = self.config.verifier.multiround_start_round
        return SimpleNamespace(stats=SimpleNamespace(evals={}))

    original_job_run = job_module.Job.run
    original_show_hint = jobs_module.show_registry_hint_if_first_run
    original_print_tables = jobs_module.print_job_results_tables

    job_module.Job.run = fake_run
    jobs_module.show_registry_hint_if_first_run = lambda console: None
    jobs_module.print_job_results_tables = lambda _: None
    try:
        start(
            path=task_dir,
            agent_name="claude-code",
            resume_trial=job_dir,
            resume_round=3,
            output_jobs_dir=tmp_path / "output-jobs",
        )
    finally:
        job_module.Job.run = original_job_run
        jobs_module.show_registry_hint_if_first_run = original_show_hint
        jobs_module.print_job_results_tables = original_print_tables

    assert captured["resume_source"] == str(selected_trial.resolve())
    assert captured["start_round"] == 3


def test_start_rejects_resume_round_greater_than_max_round(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=6)

    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=task_dir,
    )

    with pytest.raises(ValueError, match="--resume-round \\(5\\) must be <= --max-round \\(3\\)"):
        start(
            path=task_dir,
            agent_name="oracle",
            resume_trial=resume_trial_dir,
            resume_round=5,
            max_round=3,
        )


def test_start_rejects_resume_round_greater_than_task_num_rounds(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=task_dir,
    )

    with pytest.raises(
        ValueError,
        match="--resume-round \\(5\\) must be <= the task's num_rounds \\(3\\)",
    ):
        start(
            path=task_dir,
            agent_name="oracle",
            resume_trial=resume_trial_dir,
            resume_round=5,
        )


def test_start_rejects_resume_trial_from_different_task(tmp_path: Path):
    current_task_dir = tmp_path / "current-task"
    resume_task_dir = tmp_path / "resume-task"
    _write_multiround_task(current_task_dir, num_rounds=3)
    _write_multiround_task(resume_task_dir, num_rounds=4)

    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=resume_task_dir,
    )
    _write_source_trial_result(
        resume_trial_dir,
        task_checksum="resume-task-checksum",
    )

    with pytest.raises(
        ValueError,
        match="--resume-trial source task does not match the current task",
    ):
        start(
            path=current_task_dir,
            agent_name="oracle",
            resume_trial=resume_trial_dir,
            resume_round=2,
        )


def test_start_rejects_resume_without_snapshot_before_backup_cleanup(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    resume_trial_dir = tmp_path / "oracle-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="oracle",
        task_path=task_dir,
    )

    with pytest.raises(
        ValueError,
        match="no matching resume snapshot metadata",
    ):
        start(
            path=task_dir,
            agent_name="oracle",
            resume_trial=resume_trial_dir,
            resume_round=2,
        )

    assert (resume_trial_dir / "config.json").exists()
    assert not list(resume_trial_dir.parent.glob("oracle-source__resumed_*"))


def test_start_rejects_resume_round_without_resume_trial(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_task(task_dir)

    with pytest.raises(ValueError, match="--resume-round requires --resume-trial"):
        start(
            path=task_dir,
            agent_name="oracle",
            resume_round=3,
        )


def test_start_rejects_disable_verification_for_multiround_task(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    with pytest.raises(
        ValueError,
        match="--disable-verification is not supported for multi-round tasks",
    ):
        start(
            path=task_dir,
            agent_name="oracle",
            disable_verification=True,
        )


def test_start_rejects_start_round_greater_than_task_num_rounds(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    with pytest.raises(
        ValueError,
        match="--start-round \\(5\\) must be <= the task's num_rounds \\(3\\)",
    ):
        start(
            path=task_dir,
            agent_name="oracle",
            start_round=5,
        )


def test_start_allows_snapshot_only_resume_preflight_without_agent_state(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    resume_trial_dir = tmp_path / "claude-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="claude-code",
        task_path=task_dir,
    )
    _write_round_snapshot_metadata(resume_trial_dir, round_num=2)

    import harbor.job as job_module
    import harbor.cli.jobs as jobs_module

    async def fake_run(self):
        return SimpleNamespace(stats=SimpleNamespace(evals={}))

    original_job_run = job_module.Job.run
    original_show_hint = jobs_module.show_registry_hint_if_first_run
    original_print_tables = jobs_module.print_job_results_tables

    job_module.Job.run = fake_run
    jobs_module.show_registry_hint_if_first_run = lambda console: None
    jobs_module.print_job_results_tables = lambda _: None
    try:
        start(
            path=task_dir,
            agent_name="claude-code",
            resume_trial=resume_trial_dir,
            resume_round=3,
            multiround_resume_preflight_policy="snapshot",
        )
    finally:
        job_module.Job.run = original_job_run
        jobs_module.show_registry_hint_if_first_run = original_show_hint
        jobs_module.print_job_results_tables = original_print_tables


def test_start_strict_resume_preflight_requires_agent_state(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir, num_rounds=3)

    resume_trial_dir = tmp_path / "claude-source"
    _write_source_trial_config(
        resume_trial_dir,
        source_agent_name="claude-code",
        task_path=task_dir,
    )
    _write_round_snapshot_metadata(resume_trial_dir, round_num=2)

    with pytest.raises(
        ValueError,
        match="no Claude session snapshot",
    ):
        start(
            path=task_dir,
            agent_name="claude-code",
            resume_trial=resume_trial_dir,
            resume_round=3,
        )


def test_start_rejects_fanout_with_cache_policy_off(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir)

    with pytest.raises(
        ValueError,
        match="--n-attempts > 1 with --multiround-state-cache-policy off",
    ):
        start(
            path=task_dir,
            agent_name="oracle",
            n_attempts=4,
            multiround_state_cache_policy="off",
        )


def test_start_allows_nonfanout_attempts_with_cache_policy_off(tmp_path: Path):
    task_dir = tmp_path / "task"
    _write_task(task_dir)

    config = JobConfig(
        n_attempts=4,
        verifier={"multiround_state_cache_policy": "off"},
        tasks=[TaskConfig(path=task_dir)],
    )

    assert (
        _would_enable_roundwise_multiround_attempt_selection(config) is False
    )

    import harbor.job as job_module
    import harbor.cli.jobs as jobs_module

    async def fake_run(self):
        return SimpleNamespace(stats=SimpleNamespace(evals={}))

    original_job_run = job_module.Job.run
    original_show_hint = jobs_module.show_registry_hint_if_first_run
    original_print_tables = jobs_module.print_job_results_tables

    job_module.Job.run = fake_run
    jobs_module.show_registry_hint_if_first_run = lambda console: None
    jobs_module.print_job_results_tables = lambda _: None
    try:
        # The CLI guard should therefore not treat this single-round task as a fanout job.
        start(
            path=task_dir,
            jobs_dir=tmp_path / "jobs",
            job_name="single-round-off",
            agent_name="oracle",
            n_attempts=4,
            multiround_state_cache_policy="off",
        )
    finally:
        job_module.Job.run = original_job_run
        jobs_module.show_registry_hint_if_first_run = original_show_hint
        jobs_module.print_job_results_tables = original_print_tables


def test_resume_source_with_multiple_attempts_enables_roundwise_attempt_selection(
    tmp_path: Path,
):
    task_dir = tmp_path / "task"
    _write_multiround_task(task_dir)

    config = JobConfig(
        jobs_dir=tmp_path / "jobs",
        job_name="resume-fanout",
        n_attempts=4,
        tasks=[TaskConfig(path=task_dir)],
    )
    config.verifier.multiround_start_round = 3
    config.verifier.multiround_resume_source = str(
        tmp_path / "source-task__abc1234__mr-r2-a0"
    )

    assert _would_enable_roundwise_multiround_attempt_selection(config) is True
