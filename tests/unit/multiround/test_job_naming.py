import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import harbor.job as job_module
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.task.id import LocalTaskId
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import AgentInfo, EnvironmentStateInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.queue import LiveTrialExecution


def _make_trial_result(
    config: TrialConfig, trial_dir: Path, round_num: int
) -> TrialResult:
    return TrialResult(
        task_name=config.task.get_task_id().get_name(),
        trial_name=config.trial_name,
        trial_uri=trial_dir.resolve().as_uri(),
        task_id=LocalTaskId(path=config.task.path),
        task_checksum="checksum",
        config=config,
        agent_info=AgentInfo(name="oracle", version="test"),
        verifier_result=VerifierResult(rewards={f"round_{round_num}": 1.0}),
        environment_state=EnvironmentStateInfo(
            snapshot_id=f"{config.trial_name}__state",
            image_ref="hbstate__test",
            image_archive=str((trial_dir / "state" / "snapshot-image.tar").resolve()),
        ),
    )


def _write_snapshot_state(trial_dir: Path, round_num: int) -> None:
    trial_paths = TrialPaths(trial_dir)
    trial_paths.mkdir()
    round_state_dir = trial_paths.round_state_dir(round_num)
    round_state_dir.mkdir(parents=True, exist_ok=True)
    round_archive = trial_paths.round_state_image_archive_path(round_num)
    round_archive.write_text(f"snapshot-{round_num}")
    trial_paths.round_state_snapshot_path(round_num).write_text("{}")
    trial_paths.state_snapshot_path.write_text("{}")
    trial_paths.state_image_archive_path.write_text(f"latest-{round_num}")


@pytest.mark.asyncio
async def test_prune_trial_snapshot_state_deletes_daytona_state_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    job = object.__new__(Job)
    job._logger = job_module.logger.getChild(
        "test_job_multiround_daytona_snapshot_cleanup"
    )

    config = TrialConfig(task=TaskConfig(path=tmp_path / "demo-task"))
    trial_dir = tmp_path / config.trial_name
    trial_paths = TrialPaths(trial_dir)
    trial_paths.mkdir()
    round_state_dir = trial_paths.round_state_dir(1)
    round_state_dir.mkdir(parents=True)
    trial_paths.round_state_snapshot_path(1).write_text(
        json.dumps(
            {
                "snapshot_id": "round-1",
                "provider": "daytona",
                "provider_state_mode": "pause_fork",
                "image_ref": "daytona-fork-source:sandbox-1",
                "daytona_sandbox_id": "sandbox-1",
                "round": 1,
            }
        )
    )
    trial_paths.state_snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_id": "round-2",
                "provider": "daytona",
                "provider_state_mode": "snapshot",
                "image_ref": "daytona-snapshot:snapshot-2",
                "daytona_snapshot_name": "snapshot-2",
                "round": 2,
            }
        )
    )

    result = _make_trial_result(config, trial_dir, round_num=1)
    trial_paths.result_path.write_text(result.model_dump_json(indent=4))

    cleanup_calls: list[tuple[set[str], set[str]]] = []

    async def fake_delete_daytona_state_refs(
        *, sandbox_refs: set[str], snapshot_refs: set[str]
    ) -> None:
        cleanup_calls.append((set(sandbox_refs), set(snapshot_refs)))

    monkeypatch.setattr(
        job,
        "_delete_daytona_state_refs",
        fake_delete_daytona_state_refs,
    )

    await job._prune_trial_snapshot_state(result)

    assert cleanup_calls == [({"sandbox-1"}, {"snapshot-2"})]
    assert not trial_paths.state_dir.exists()


@pytest.mark.asyncio
async def test_prune_trial_snapshot_state_deletes_harbor_snapshot_image_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    job = object.__new__(Job)
    job._logger = job_module.logger.getChild("test_job_multiround_snapshot_cleanup")

    config = TrialConfig(task=TaskConfig(path=tmp_path / "demo-task"))
    trial_dir = tmp_path / config.trial_name
    trial_paths = TrialPaths(trial_dir)
    trial_paths.mkdir()
    round_state_dir = trial_paths.round_state_dir(1)
    round_state_dir.mkdir(parents=True)
    snapshot_payload = {
        "snapshot_id": "round-1",
        "image_tag": "hbstate__demo_round_1",
        "image_ref": "sha256:demo",
        "round": 1,
    }
    trial_paths.round_state_snapshot_path(1).write_text(
        json.dumps(snapshot_payload)
    )
    trial_paths.state_snapshot_path.write_text(
        json.dumps({**snapshot_payload, "image_tag": "ubuntu:latest"})
    )

    result = _make_trial_result(config, trial_dir, round_num=1)
    trial_paths.result_path.write_text(result.model_dump_json(indent=4))

    deleted_tags: list[set[str]] = []
    monkeypatch.setattr(
        job,
        "_delete_snapshot_image_tags",
        lambda image_tags: deleted_tags.append(set(image_tags)),
    )

    await job._prune_trial_snapshot_state(result)

    assert deleted_tags == [{"hbstate__demo_round_1"}]
    assert not trial_paths.state_dir.exists()
    assert result.environment_state is not None
    assert result.environment_state.image_archive is None
    persisted = TrialResult.model_validate_json(trial_paths.result_path.read_text())
    assert persisted.environment_state is not None
    assert persisted.environment_state.image_archive is None


def test_extract_multiround_attempt_idx_handles_root_and_child_names():
    assert Job._extract_multiround_attempt_idx("task__abc1234__mr-r1-a3") == 3
    assert (
        Job._extract_multiround_attempt_idx("task__abc1234__mr-r2-p7-hZX81aa3-a11")
        == 11
    )
    assert Job._extract_multiround_attempt_idx("task__abc1234") is None


@pytest.mark.asyncio
async def test_roundwise_multiround_child_names_use_parent_attempt_idx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        job_module,
        "Task",
        lambda path: SimpleNamespace(name=path.name, num_rounds=2),
    )

    job = object.__new__(Job)
    job.config = JobConfig(n_attempts=4)
    job.config.verifier.multiround_continue_successes_per_round = 1
    job.config.verifier.multiround_state_cache_policy = "all"
    job.config.verifier.multiround_state_retention_policy = "selected"
    job._remaining_trial_configs = [
        TrialConfig(task=TaskConfig(path=tmp_path / "demo-task")) for _ in range(4)
    ]
    job._logger = job_module.logger.getChild("test_job_multiround_naming")

    captured_frontiers: list[list[str]] = []

    async def fake_run_trial_batch(frontier: list[TrialConfig]) -> list[TrialResult]:
        captured_frontiers.append([config.trial_name for config in frontier])
        if len(captured_frontiers) == 1:
            selected_parent = frontier[3]
            parent_dir = tmp_path / "parent-round-1"
            parent_dir.mkdir()
            return [_make_trial_result(selected_parent, parent_dir, round_num=1)]

        return [
            _make_trial_result(config, tmp_path / config.trial_name, round_num=2)
            for config in frontier
        ]

    job._run_trial_batch = fake_run_trial_batch

    await job._run_roundwise_multiround_attempt_selection()

    assert len(captured_frontiers) == 2
    suffixes = [name.split("__mr-r2-", 1)[1] for name in captured_frontiers[1]]
    assert suffixes == sorted(suffixes)
    assert all(re.fullmatch(r"p3-h[A-Za-z0-9]+-a\d+", suffix) for suffix in suffixes)
    assert [suffix.rsplit("-a", 1)[1] for suffix in suffixes] == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_roundwise_multiround_single_attempt_uses_same_frontier_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        job_module,
        "Task",
        lambda path: SimpleNamespace(name=path.name, num_rounds=2),
    )

    job = object.__new__(Job)
    job.config = JobConfig(n_attempts=1)
    job.config.verifier.multiround_continue_successes_per_round = 1
    job.config.verifier.multiround_state_cache_policy = "all"
    job.config.verifier.multiround_state_retention_policy = "selected"
    job._remaining_trial_configs = [
        TrialConfig(task=TaskConfig(path=tmp_path / "demo-task"))
    ]
    job._logger = job_module.logger.getChild("test_job_multiround_single_attempt")

    captured_frontiers: list[list[TrialConfig]] = []

    async def fake_run_trial_batch(frontier: list[TrialConfig]) -> list[TrialResult]:
        captured_frontiers.append(frontier)
        round_num = len(captured_frontiers)
        parent_dir = tmp_path / f"round-{round_num}-trial"
        parent_dir.mkdir()
        return [_make_trial_result(frontier[0], parent_dir, round_num=round_num)]

    job._run_trial_batch = fake_run_trial_batch

    await job._run_roundwise_multiround_attempt_selection()

    assert len(captured_frontiers) == 2
    assert len(captured_frontiers[0]) == 1
    assert len(captured_frontiers[1]) == 1
    assert "__mr-r1-a0" in captured_frontiers[0][0].trial_name
    assert "__mr-r2-p0-" in captured_frontiers[1][0].trial_name
    assert captured_frontiers[1][0].trial_name.endswith("-a0")
    assert captured_frontiers[1][0].verifier.multiround_start_round == 2
    assert captured_frontiers[1][0].verifier.multiround_max_round == 2
    assert captured_frontiers[1][0].verifier.multiround_resume_source is not None


@pytest.mark.asyncio
async def test_roundwise_multiround_resume_frontier_uses_resume_parent_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        job_module,
        "Task",
        lambda path: SimpleNamespace(name=path.name, num_rounds=3),
    )

    job = object.__new__(Job)
    job.config = JobConfig(n_attempts=4)
    job.config.verifier.multiround_continue_successes_per_round = 1
    job.config.verifier.multiround_state_cache_policy = "all"
    job.config.verifier.multiround_state_retention_policy = "selected"
    job._remaining_trial_configs = [
        TrialConfig(
            task=TaskConfig(path=tmp_path / "demo-task"),
            verifier={
                "multiround_start_round": 3,
                "multiround_resume_source": str(
                    tmp_path / "demo-task__abc1234__mr-r2-p7-hZX81aa3-a11"
                ),
            },
        )
        for _ in range(4)
    ]
    job._logger = job_module.logger.getChild("test_job_multiround_resume_naming")

    captured_frontiers: list[list[str]] = []

    async def fake_run_trial_batch(frontier: list[TrialConfig]) -> list[TrialResult]:
        captured_frontiers.append([config.trial_name for config in frontier])
        return [
            _make_trial_result(config, tmp_path / config.trial_name, round_num=3)
            for config in frontier
        ]

    job._run_trial_batch = fake_run_trial_batch

    await job._run_roundwise_multiround_attempt_selection()

    assert len(captured_frontiers) == 1
    suffixes = [name.split("__mr-r3-", 1)[1] for name in captured_frontiers[0]]
    assert all(re.fullmatch(r"p11-habc1234-a\d+", suffix) for suffix in suffixes)
    assert [suffix.rsplit("-a", 1)[1] for suffix in suffixes] == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_roundwise_multiround_prunes_unselected_snapshot_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        job_module,
        "Task",
        lambda path: SimpleNamespace(name=path.name, num_rounds=2),
    )

    job = object.__new__(Job)
    job.config = JobConfig(n_attempts=3)
    job.config.verifier.multiround_continue_successes_per_round = 1
    job.config.verifier.multiround_state_cache_policy = "all"
    job.config.verifier.multiround_state_retention_policy = "selected"
    job._remaining_trial_configs = [
        TrialConfig(task=TaskConfig(path=tmp_path / "demo-task")) for _ in range(3)
    ]
    job._logger = job_module.logger.getChild("test_job_multiround_snapshot_pruning")

    first_round_results: list[TrialResult] = []

    async def fake_run_trial_batch(frontier: list[TrialConfig]) -> list[TrialResult]:
        if not first_round_results:
            for idx, config in enumerate(frontier):
                trial_dir = tmp_path / config.trial_name
                _write_snapshot_state(trial_dir, round_num=1)
                if idx < 2:
                    first_round_results.append(
                        _make_trial_result(config, trial_dir, round_num=1)
                    )
                else:
                    first_round_results.append(
                        TrialResult(
                            task_name=config.task.get_task_id().get_name(),
                            trial_name=config.trial_name,
                            trial_uri=trial_dir.resolve().as_uri(),
                            task_id=LocalTaskId(path=config.task.path),
                            task_checksum="checksum",
                            config=config,
                            agent_info=AgentInfo(name="oracle", version="test"),
                            verifier_result=VerifierResult(rewards={"round_1": 0.0}),
                            environment_state=EnvironmentStateInfo(
                                snapshot_id=f"{config.trial_name}__state",
                                image_ref="hbstate__test",
                                image_archive=str(
                                    (
                                        trial_dir / "state" / "snapshot-image.tar"
                                    ).resolve()
                                ),
                            ),
                        )
                    )
            return list(first_round_results)

        return [
            _make_trial_result(config, tmp_path / config.trial_name, round_num=2)
            for config in frontier
        ]

    job._run_trial_batch = fake_run_trial_batch

    await job._run_roundwise_multiround_attempt_selection()

    kept_parent = min(first_round_results[:2], key=lambda result: result.trial_name)
    pruned_success = next(
        result
        for result in first_round_results[:2]
        if result.trial_name != kept_parent.trial_name
    )
    pruned_failure = first_round_results[2]

    assert (Path(urlparse(kept_parent.trial_uri).path) / "state").exists()
    assert not (Path(urlparse(pruned_success.trial_uri).path) / "state").exists()
    assert not (Path(urlparse(pruned_failure.trial_uri).path) / "state").exists()
    assert kept_parent.environment_state is not None
    assert kept_parent.environment_state.image_archive is not None
    assert pruned_success.environment_state is not None
    assert pruned_success.environment_state.image_archive is None
    assert pruned_failure.environment_state is not None
    assert pruned_failure.environment_state.image_archive is None


@pytest.mark.asyncio
async def test_roundwise_success_policy_captures_only_selected_parent_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        job_module,
        "Task",
        lambda path: SimpleNamespace(name=path.name, num_rounds=2),
    )

    job = object.__new__(Job)
    job.config = JobConfig(n_attempts=3)
    job.config.verifier.multiround_continue_successes_per_round = 1
    job.config.verifier.multiround_state_cache_policy = "success"
    job._remaining_trial_configs = [
        TrialConfig(task=TaskConfig(path=tmp_path / "demo-task")) for _ in range(3)
    ]
    job._logger = job_module.logger.getChild(
        "test_job_multiround_selected_only_snapshot"
    )

    capture_calls: list[tuple[str, int]] = []
    prune_calls: list[str] = []
    stop_calls: list[str] = []
    live_batch_flags: list[tuple[bool, bool]] = []
    captured_frontiers: list[list[TrialConfig]] = []

    class FakeLiveTrial:
        def __init__(self, result: TrialResult):
            self.result = result

        async def capture_multiround_state_snapshot(
            self, *, round_num: int, restart_environment: bool
        ) -> None:
            capture_calls.append((self.result.trial_name, round_num))
            assert restart_environment is False
            self.result.environment_state = EnvironmentStateInfo(
                snapshot_id=f"{self.result.trial_name}__round-{round_num}__state",
                image_ref=f"hbstate__{self.result.trial_name}__round_{round_num}",
                image_archive=str(
                    (
                        Path(urlparse(self.result.trial_uri).path)
                        / "state"
                        / f"round_{round_num}"
                        / "snapshot-image.tar"
                    ).resolve()
                ),
            )

        async def stop_agent_environment(self) -> None:
            stop_calls.append(self.result.trial_name)

    async def fake_run_live_trial_batch(
        frontier: list[TrialConfig],
        *,
        keep_environment_alive_on_success: bool,
        defer_multiround_state_snapshot: bool,
    ) -> list[LiveTrialExecution]:
        live_batch_flags.append(
            (keep_environment_alive_on_success, defer_multiround_state_snapshot)
        )
        captured_frontiers.append(frontier)
        round_num = len(captured_frontiers)
        executions: list[LiveTrialExecution] = []
        for idx, config in enumerate(frontier):
            trial_dir = tmp_path / config.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            reward = 1.0 if round_num == 2 or idx < 2 else 0.0
            result = TrialResult(
                task_name=config.task.get_task_id().get_name(),
                trial_name=config.trial_name,
                trial_uri=trial_dir.resolve().as_uri(),
                task_id=LocalTaskId(path=config.task.path),
                task_checksum="checksum",
                config=config,
                agent_info=AgentInfo(name="oracle", version="test"),
                verifier_result=VerifierResult(rewards={f"round_{round_num}": reward}),
            )
            executions.append(
                LiveTrialExecution(trial=FakeLiveTrial(result), result=result)
            )
        return executions

    async def fake_prune_trial_snapshot_state(result: TrialResult) -> None:
        prune_calls.append(result.trial_name)
        if result.environment_state is not None:
            result.environment_state.image_archive = None

    job._run_live_trial_batch = fake_run_live_trial_batch
    job._prune_trial_snapshot_state = fake_prune_trial_snapshot_state

    await job._run_roundwise_multiround_attempt_selection()

    assert live_batch_flags == [(True, True), (True, True)]
    assert len(captured_frontiers) == 2
    first_round_successes = captured_frontiers[0][:2]
    expected_round_1_parent = min(
        config.trial_name for config in first_round_successes
    )
    expected_round_2_parent = min(config.trial_name for config in captured_frontiers[1])
    assert capture_calls == [
        (expected_round_1_parent, 1),
        (expected_round_2_parent, 2),
    ]
    assert prune_calls[-1] == expected_round_1_parent
    assert sorted(stop_calls) == sorted(
        config.trial_name for frontier in captured_frontiers for config in frontier
    )
    assert all(
        config.verifier.multiround_resume_state_image
        == f"hbstate__{expected_round_1_parent}__round_1"
        for config in captured_frontiers[1]
    )
