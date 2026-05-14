import hashlib
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig
from harbor.models.task.paths import TaskPaths


_VOLATILE_TASK_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "agent_logs",
    "eval_runs",
    "harbor_jobs",
    "log",
    "logs",
}
_VOLATILE_TASK_FILE_NAMES = {
    ".env",
    "conditionally_passed.txt",
    "creating_jobs_latest.txt",
    "failed.txt",
    "passed.txt",
}


def _is_volatile_task_artifact(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part.startswith("creating_"):
            return True
        if part in _VOLATILE_TASK_DIR_NAMES:
            return True

    return relative_path.name in _VOLATILE_TASK_FILE_NAMES


class Task:
    _ALLOWED_CHANGE_TYPES = {"extension", "correction", "conflict"}
    """
    Represents a task with the following directory structure:

    ├── instruction.md
    ├── task.toml
    ├── environment/
    │   ├── [docker-compose.yaml | Dockerfile | singularity-compose.yaml | etc.]
    │   └── ...
    ├── solution/         # copied to container @ /solution by OracleAgent
    │   ├── solve.sh
    │   └── ...
    └── tests/            # copied to container @ /tests by Evaluator
        ├── test.sh
        └── ...
    """

    def __init__(self, task_dir: Path | str):
        """
        Initialize a Task from a directory path.

        Args:
            task_dir: Path to the task directory
        """
        self._task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self._task_dir)
        self.instruction = self.paths.instruction_path.read_text()
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text())
        self.name = self.paths.task_dir.name
        self._validate_multiround_definition()

    @property
    def checksum(self) -> str:
        """Generate a deterministic hash for the task definition.

        Runtime artifacts live under task directories during large eval rollouts.
        Excluding them keeps checksums stable while logs are being written.
        """
        digest = hashlib.sha256()

        for path in sorted(self._task_dir.rglob("*")):
            if not path.is_file():
                continue

            relative_path = path.relative_to(self._task_dir)
            if _is_volatile_task_artifact(relative_path):
                continue

            digest.update(relative_path.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")

        return digest.hexdigest()

    @property
    def task_dir(self) -> Path:
        """Public accessor for the task directory."""
        return self._task_dir

    # ---- Multi-round helpers ----

    @property
    def is_multiround(self) -> bool:
        """Check if this task is a multi-round task."""
        multiround = self.config.metadata.get("multiround")
        return multiround is not None and multiround.get("num_rounds", 0) > 1

    @property
    def num_rounds(self) -> int:
        """Number of rounds. Returns 1 for single-round tasks."""
        if not self.is_multiround:
            return 1
        return self.config.metadata["multiround"]["num_rounds"]

    @property
    def round_configs(self) -> list[dict]:
        """Per-round metadata with normalized change type fields.

        Supports both legacy ``change_type = "extension"`` and the new
        composable ``change_types = ["extension", "correction"]`` schema.
        """
        if not self.is_multiround:
            return []
        return [
            self._normalize_round_config(round_config)
            for round_config in self.config.metadata["multiround"].get("rounds", [])
        ]

    def round_instruction(self, round_num: int) -> str:
        """Read instruction for a specific round."""
        return self.paths.round_instruction_path(round_num).read_text()

    def round_config(self, round_num: int) -> dict[str, Any]:
        """Per-round metadata for the requested round."""
        return next(
            (rc for rc in self.round_configs if rc.get("round") == round_num),
            {},
        )

    def round_change_types(self, round_num: int) -> list[str]:
        """Normalized change type tags for the requested round."""
        return self.round_config(round_num).get("change_types", [])

    def round_change_type_label(self, round_num: int) -> str:
        """Human-readable change type label for logs/results."""
        return self.round_config(round_num).get("change_type", "unknown")

    @classmethod
    def _normalize_round_config(cls, round_config: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(round_config)

        raw_change_types: Any = normalized.get("change_types")
        if raw_change_types is None and "change_type" in normalized:
            raw_change_types = normalized.get("change_type")

        change_types = cls._normalize_change_types(raw_change_types)
        normalized["change_types"] = change_types
        normalized["change_type"] = (
            "+".join(change_types) if change_types else "unknown"
        )
        return normalized

    def _validate_multiround_definition(self) -> None:
        if not self.is_multiround:
            return

        num_rounds = self.num_rounds
        round_configs = self.config.metadata["multiround"].get("rounds", [])
        actual_round_numbers = self.paths.round_numbers()

        if actual_round_numbers != list(range(1, num_rounds + 1)):
            raise ValueError(
                "Multi-round task directories must be exactly round_1..round_"
                f"{num_rounds}; found {actual_round_numbers or 'none'}"
            )

        if len(round_configs) != num_rounds:
            raise ValueError(
                "metadata.multiround.rounds must contain one entry per round; "
                f"expected {num_rounds}, got {len(round_configs)}"
            )

        seen_rounds: list[int] = []
        for index, raw_round_config in enumerate(round_configs, start=1):
            round_num = raw_round_config.get("round")
            if round_num != index:
                raise ValueError(
                    "metadata.multiround.rounds must use contiguous round numbers "
                    f"starting at 1; expected round={index}, got {round_num}"
                )

            normalized = self._normalize_round_config(raw_round_config)
            change_types = normalized["change_types"]
            if not change_types:
                raise ValueError(
                    f"Round {round_num} must define a non-empty change_types list"
                )

            raw_change_types = raw_round_config.get("change_types")
            if raw_change_types is None and "change_type" in raw_round_config:
                raw_change_types = [raw_round_config["change_type"]]
            if isinstance(raw_change_types, list) and len(raw_change_types) != len(
                change_types
            ):
                raise ValueError(
                    f"Round {round_num} change_types must not contain duplicates"
                )

            if not self.paths.round_instruction_path(round_num).exists():
                raise ValueError(
                    f"Round {round_num} is missing instruction.md at "
                    f"{self.paths.round_instruction_path(round_num)}"
                )
            if not self.paths.round_solve_path(round_num).exists():
                raise ValueError(
                    f"Round {round_num} is missing solution/solve.sh at "
                    f"{self.paths.round_solve_path(round_num)}"
                )
            if not self.paths.round_test_path(round_num).exists():
                raise ValueError(
                    f"Round {round_num} is missing tests/test.sh at "
                    f"{self.paths.round_test_path(round_num)}"
                )

            seen_rounds.append(round_num)

        if seen_rounds != list(range(1, num_rounds + 1)):
            raise ValueError(
                "metadata.multiround.rounds must cover every round from 1 to "
                f"{num_rounds}; found {seen_rounds}"
            )

    @classmethod
    def _normalize_change_types(cls, raw_change_types: Any) -> list[str]:
        if raw_change_types is None:
            return []

        if isinstance(raw_change_types, str):
            candidates = [raw_change_types]
        elif isinstance(raw_change_types, list):
            candidates = raw_change_types
        else:
            raise ValueError(
                "Round change types must be a string or list of strings, got "
                f"{type(raw_change_types).__name__}"
            )

        normalized: list[str] = []
        seen: set[str] = set()
        for value in candidates:
            if not isinstance(value, str):
                raise ValueError(
                    "Round change types must contain only strings, got "
                    f"{type(value).__name__}"
                )
            value = value.strip()
            if not value:
                raise ValueError("Round change types must not contain empty strings")
            if value not in cls._ALLOWED_CHANGE_TYPES:
                allowed = ", ".join(sorted(cls._ALLOWED_CHANGE_TYPES))
                raise ValueError(
                    f"Unsupported round change type '{value}'. Allowed: {allowed}"
                )
            if value in seen:
                raise ValueError(f"Duplicate round change type '{value}'")
            normalized.append(value)
            seen.add(value)

        return normalized
