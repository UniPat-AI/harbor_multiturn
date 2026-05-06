from pathlib import Path
from types import SimpleNamespace

from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.orchestrators.local import LocalOrchestrator


def _result(*, reward=None, has_verifier: bool = True, has_exception: bool = False):
    verifier_result = None
    if has_verifier:
        verifier_result = SimpleNamespace(
            rewards=None if reward is None else {"reward": reward}
        )
    exception_info = SimpleNamespace(exception_type="Boom") if has_exception else None
    return SimpleNamespace(
        verifier_result=verifier_result,
        exception_info=exception_info,
    )


def test_trial_progress_status_marks_exception_as_error():
    assert LocalOrchestrator._trial_progress_status(
        _result(reward=1.0, has_exception=True)
    ) == "error"


def test_trial_progress_status_marks_zero_reward_as_failed():
    assert LocalOrchestrator._trial_progress_status(_result(reward=0.0)) == "failed"


def test_trial_progress_status_keeps_positive_reward_as_ok():
    assert LocalOrchestrator._trial_progress_status(_result(reward=1.0)) == "ok"


def test_trial_progress_status_keeps_missing_verifier_reward_as_ok():
    assert LocalOrchestrator._trial_progress_status(
        _result(reward=None, has_verifier=False)
    ) == "ok"


def _write_single_round_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("solve it")
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")


def _write_multiround_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("solve it")
    (task_dir / "task.toml").write_text(
        '\n'.join(
            [
                'version = "1.0"',
                "",
                "[metadata.multiround]",
                "num_rounds = 2",
                "",
                "[[metadata.multiround.rounds]]",
                "round = 1",
                'change_type = "extension"',
                "",
                "[[metadata.multiround.rounds]]",
                "round = 2",
                'change_type = "correction"',
                "",
            ]
        )
    )
    for round_num in (1, 2):
        round_tests_dir = task_dir / f"round_{round_num}" / "tests"
        round_solution_dir = task_dir / f"round_{round_num}" / "solution"
        round_tests_dir.mkdir(parents=True, exist_ok=True)
        round_solution_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"round_{round_num}" / "instruction.md").write_text(
            f"round {round_num}"
        )
        (round_tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n")
        (round_solution_dir / "solve.sh").write_text("#!/bin/sh\nexit 0\n")


def _orchestrator_for_task(task_dir: Path) -> LocalOrchestrator:
    return LocalOrchestrator(
        trial_configs=[TrialConfig(task={"path": task_dir})],
        n_concurrent_trials=1,
        metrics={},
        quiet=True,
        retry_config=RetryConfig(),
    )


def test_should_use_plain_quiet_progress_only_for_multiround_tasks(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("HARBOR_PLAIN_QUIET_PROGRESS", "true")

    single_task_dir = tmp_path / "single"
    _write_single_round_task(single_task_dir)
    assert _orchestrator_for_task(single_task_dir)._should_use_plain_quiet_progress() is False

    multiround_task_dir = tmp_path / "multi"
    _write_multiround_task(multiround_task_dir)
    assert (
        _orchestrator_for_task(multiround_task_dir)._should_use_plain_quiet_progress()
        is True
    )
