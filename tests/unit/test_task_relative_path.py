from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task


def test_task_init_with_dot_path(tmp_path, monkeypatch):
    # Create minimal valid task structure in a temporary directory
    task_dir = tmp_path / "my-task"
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"

    env_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    # Minimal Dockerfile so environment validation would pass if used
    (env_dir / "Dockerfile").write_text("FROM alpine:3.19\n")

    # Minimal test script presence
    (tests_dir / "test.sh").write_text("#!/usr/bin/env sh\nexit 0\n")

    # Write required files
    (task_dir / "instruction.md").write_text("Do something simple.\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[environment]
""".strip()
    )

    assert TaskPaths(task_dir).is_valid() is True

    # Change working directory to the task directory
    monkeypatch.chdir(task_dir)

    # Initialize Task using relative path '.'
    task = Task(task_dir=".")

    # Assert paths are resolved to absolute and name is correct
    assert task.task_dir == task_dir.resolve()
    assert task.paths.task_dir == task_dir.resolve()
    assert task.name == task_dir.name


def test_task_paths_reject_round_tests_without_multiround_metadata(tmp_path):
    task_dir = tmp_path / "broken-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Task\n")
    (task_dir / "task.toml").write_text('version = "1.0"\n')
    round_tests_dir = task_dir / "round_1" / "tests"
    round_tests_dir.mkdir(parents=True)
    (round_tests_dir / "test.sh").write_text("#!/usr/bin/env sh\nexit 0\n")

    assert TaskPaths(task_dir).is_valid() is False


def test_task_paths_accept_declared_multiround_task(tmp_path):
    task_dir = tmp_path / "multi-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Task\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[metadata.multiround]
num_rounds = 2
""".strip()
    )

    for round_num in (1, 2):
        round_dir = task_dir / f"round_{round_num}"
        (round_dir / "solution").mkdir(parents=True)
        (round_dir / "tests").mkdir(parents=True)
        (round_dir / "instruction.md").write_text(f"Round {round_num}\n")
        (round_dir / "solution" / "solve.sh").write_text("#!/usr/bin/env sh\n")
        (round_dir / "tests" / "test.sh").write_text("#!/usr/bin/env sh\n")

    assert TaskPaths(task_dir).is_valid() is True


def test_task_paths_accept_in_progress_one_round_multiround_task(tmp_path):
    task_dir = tmp_path / "multi-task-one-round"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Task\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[metadata.multiround]
num_rounds = 1

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension"]
""".strip()
    )

    round_dir = task_dir / "round_1"
    (round_dir / "solution").mkdir(parents=True)
    (round_dir / "tests").mkdir(parents=True)
    (round_dir / "instruction.md").write_text("Round 1\n")
    (round_dir / "solution" / "solve.sh").write_text("#!/usr/bin/env sh\n")
    (round_dir / "tests" / "test.sh").write_text("#!/usr/bin/env sh\n")

    assert TaskPaths(task_dir).is_valid() is True
    assert Task(task_dir).is_multiround is True
    assert Task(task_dir).num_rounds == 1
