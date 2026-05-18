import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harbor.models.task.config import TaskConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config

# Matches canary lines: HTML comments (<!-- ...canary... -->) or hash comments (# ...canary...)
_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def strip_canary(text: str) -> str:
    """Strip canary marker lines from the beginning of instruction text.

    Canary strings are data-provenance markers embedded in benchmark files to prevent
    training-data contamination. They should not be shown to agents. This function
    removes leading lines that match common canary comment formats (HTML comments or
    hash comments containing the word "canary"), plus any blank lines that follow.
    """
    lines = text.split("\n")
    idx = 0
    # Remove leading canary comment lines
    while idx < len(lines) and _CANARY_LINE_RE.match(lines[idx].strip()):
        idx += 1
    # Remove any blank lines immediately after the canary block
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return "\n".join(lines[idx:])


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
    │   ├── solve.{sh,ps1,cmd,bat}
    │   └── ...
    └── tests/            # copied to container @ /tests by Evaluator
        ├── test.{sh,ps1,cmd,bat}
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
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text())
        if self.config.task is not None:
            self.name = self.config.task.name
        else:
            self.name = self.paths.task_dir.name

        self._validate_multiround_shape()
        self._validate_tests(self.config, self.paths)
        if self.has_steps:
            self.instruction = ""
        else:
            self.instruction = strip_canary(self.paths.instruction_path.read_text())
        self._validate_multiround_definition()

    @staticmethod
    def is_valid_dir(
        task_dir: Path | str,
        disable_verification: bool = False,
    ) -> bool:
        """Return whether a directory is a valid task directory.

        This is the config-aware task validation entrypoint. ``TaskPaths``
        only knows filesystem conventions; this method owns checks that need
        the parsed task config, such as effective verifier OS selection.
        """
        paths = TaskPaths(task_dir)
        if not paths.config_path.exists() or not paths.environment_dir.exists():
            return False

        try:
            config = TaskConfig.model_validate_toml(paths.config_path.read_text())
        except (OSError, tomllib.TOMLDecodeError, ValidationError):
            return False

        if disable_verification:
            if config.steps:
                return all(
                    paths.step_dir(step.name).exists()
                    and paths.step_instruction_path(step.name).exists()
                    for step in config.steps
                )
            return paths.instruction_path.exists()

        try:
            Task._validate_tests(config, paths)
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _validate_tests(config: TaskConfig, paths: TaskPaths) -> None:
        """Raise FileNotFoundError if host artifacts required at runtime are missing."""
        if Task._is_multiround_config(config):
            if not paths.instruction_path.exists():
                raise FileNotFoundError(
                    f"Task directory {paths.task_dir} is missing instruction.md."
                )
            verifier_env = resolve_effective_verifier_env_config(config, step_cfg=None)
            if verifier_env is not None:
                return
            verifier_os = config.environment.os
            num_rounds = Task._multiround_num_rounds(config)
            for round_num in range(1, num_rounds + 1):
                if paths.discovered_round_test_path_for(round_num, verifier_os) is None:
                    expected = paths.round_test_path_for(
                        round_num, verifier_os
                    ).relative_to(paths.task_dir)
                    raise FileNotFoundError(
                        f"Task directory {paths.task_dir} declares [environment].os = "
                        f"{verifier_os.value!r} but does not contain "
                        f"{expected.as_posix()}."
                    )
            return

        if not config.steps:
            if not paths.instruction_path.exists():
                raise FileNotFoundError(
                    f"Task directory {paths.task_dir} is missing instruction.md."
                )
            verifier_env = resolve_effective_verifier_env_config(config, step_cfg=None)
            if verifier_env is not None:
                return
            verifier_os = config.environment.os
            if paths.discovered_test_path_for(verifier_os) is None:
                expected = paths.test_path_for(verifier_os).relative_to(paths.task_dir)
                raise FileNotFoundError(
                    f"Task directory {paths.task_dir} declares [environment].os = "
                    f"{verifier_os.value!r} but does not contain "
                    f"{expected.as_posix()}."
                )
            return

        for step_cfg in config.steps:
            step_dir = paths.step_dir(step_cfg.name)
            if not step_dir.exists():
                raise FileNotFoundError(f"Step directory not found: {step_dir}")
            instruction = paths.step_instruction_path(step_cfg.name)
            if not instruction.exists():
                raise FileNotFoundError(f"Step instruction not found: {instruction}")

            verifier_env = resolve_effective_verifier_env_config(config, step_cfg)
            if verifier_env is not None:
                continue

            verifier_os = config.environment.os
            step_test = paths.discovered_step_test_path_for(step_cfg.name, verifier_os)
            shared_test = paths.discovered_test_path_for(verifier_os)
            if step_test is None and shared_test is None:
                expected_step = paths.step_test_path_for(
                    step_cfg.name, verifier_os
                ).relative_to(paths.task_dir)
                expected_shared = paths.test_path_for(verifier_os).relative_to(
                    paths.task_dir
                )
                raise FileNotFoundError(
                    f"Task directory {paths.task_dir} declares [environment].os = "
                    f"{config.environment.os.value!r} but step {step_cfg.name!r} "
                    "does not contain an OS-compatible test script. Expected "
                    f"{expected_step.as_posix()} or {expected_shared.as_posix()}."
                )

    @property
    def has_steps(self) -> bool:
        return bool(self.config.steps)

    def step_instruction(self, step_name: str) -> str:
        path = self.paths.step_instruction_path(step_name)
        return strip_canary(path.read_text())

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

    @staticmethod
    def _is_multiround_config(config: TaskConfig) -> bool:
        metadata = config.metadata or {}
        multiround = metadata.get("multiround")
        return isinstance(multiround, dict) and multiround.get("num_rounds", 0) >= 1

    @staticmethod
    def _multiround_num_rounds(config: TaskConfig) -> int:
        if not Task._is_multiround_config(config):
            return 1
        return int(config.metadata["multiround"]["num_rounds"])

    @property
    def is_multiround(self) -> bool:
        """Check if this task is a multi-round task."""
        return self._is_multiround_config(self.config)

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
        return strip_canary(self.paths.round_instruction_path(round_num).read_text())

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
        self._validate_multiround_shape()

        for raw_round_config in round_configs:
            round_num = raw_round_config.get("round")
            if not self.paths.round_instruction_path(round_num).exists():
                raise ValueError(
                    f"Round {round_num} is missing instruction.md at "
                    f"{self.paths.round_instruction_path(round_num)}"
                )
            task_os = self.config.environment.os
            round_solve_path = self.paths.discovered_round_solve_path_for(
                round_num, task_os
            )
            if round_solve_path is None:
                expected = self.paths.round_solve_path_for(round_num, task_os)
                raise ValueError(
                    f"Round {round_num} is missing an OS-compatible solve script at "
                    f"{expected}"
                )
            round_test_path = self.paths.discovered_round_test_path_for(
                round_num, task_os
            )
            if round_test_path is None:
                expected = self.paths.round_test_path_for(round_num, task_os)
                raise ValueError(
                    f"Round {round_num} is missing an OS-compatible test script at "
                    f"{expected}"
                )

    def _validate_multiround_shape(self) -> None:
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

        self._validate_multiround_round_metadata(round_configs)

    def _validate_multiround_round_metadata(
        self, round_configs: list[dict[str, Any]]
    ) -> None:
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

            seen_rounds.append(round_num)

        num_rounds = self.num_rounds
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
