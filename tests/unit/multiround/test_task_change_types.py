from harbor.models.task.task import Task


def _write_task(tmp_path, task_toml: str, *, num_rounds: int = 2):
    task_dir = tmp_path / "my-multiround-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text(
        "This is a multi-round task. See round_N/instruction.md.\n"
    )
    (task_dir / "task.toml").write_text(task_toml.strip() + "\n")
    for round_num in range(1, num_rounds + 1):
        round_dir = task_dir / f"round_{round_num}"
        (round_dir / "solution").mkdir(parents=True)
        (round_dir / "tests").mkdir(parents=True)
        (round_dir / "instruction.md").write_text(f"Round {round_num}\n")
        (round_dir / "solution" / "solve.sh").write_text("#!/bin/sh\n")
        (round_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    return task_dir


def test_task_normalizes_legacy_change_type_field(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 2

[[metadata.multiround.rounds]]
round = 1
change_type = "extension"

[[metadata.multiround.rounds]]
round = 2
change_type = "conflict"
""",
    )

    task = Task(task_dir)

    assert task.round_change_types(1) == ["extension"]
    assert task.round_change_type_label(1) == "extension"
    assert task.round_change_types(2) == ["conflict"]
    assert task.round_change_type_label(2) == "conflict"


def test_task_supports_composable_change_types_field(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 2

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension"]

[[metadata.multiround.rounds]]
round = 2
change_types = ["extension", "correction"]
""",
    )

    task = Task(task_dir)

    assert task.round_change_types(1) == ["extension"]
    assert task.round_change_type_label(1) == "extension"
    assert task.round_change_types(2) == ["extension", "correction"]
    assert task.round_change_type_label(2) == "extension+correction"


def test_task_rejects_duplicate_change_types(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 2

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension", "extension"]

[[metadata.multiround.rounds]]
round = 2
change_types = ["conflict"]
""",
    )

    try:
        Task(task_dir)
    except ValueError as exc:
        assert "Duplicate round change type" in str(exc)
    else:
        raise AssertionError("Expected duplicate change_types to be rejected")


def test_task_requires_round_metadata_for_every_round(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 3

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension"]

[[metadata.multiround.rounds]]
round = 2
change_types = ["correction"]
""",
        num_rounds=3,
    )

    try:
        Task(task_dir)
    except ValueError as exc:
        assert "must contain one entry per round" in str(exc)
    else:
        raise AssertionError("Expected missing round metadata to be rejected")


def test_task_requires_round_dirs_to_match_num_rounds(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 3

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension"]

[[metadata.multiround.rounds]]
round = 2
change_types = ["correction"]

[[metadata.multiround.rounds]]
round = 3
change_types = ["conflict"]
""",
        num_rounds=2,
    )

    try:
        Task(task_dir)
    except ValueError as exc:
        assert "must be exactly round_1..round_3" in str(exc)
    else:
        raise AssertionError("Expected mismatched round directories to be rejected")


def test_task_requires_non_empty_change_types(tmp_path):
    task_dir = _write_task(
        tmp_path,
        """
version = "1.0"

[metadata.multiround]
num_rounds = 2

[[metadata.multiround.rounds]]
round = 1
change_types = []

[[metadata.multiround.rounds]]
round = 2
change_types = ["extension"]
""",
    )

    try:
        Task(task_dir)
    except ValueError as exc:
        assert "non-empty change_types" in str(exc)
    else:
        raise AssertionError("Expected empty change_types to be rejected")
