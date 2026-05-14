import asyncio
import hashlib
import json
import re
import shutil
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated

import yaml
from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from harbor.agents.factory import AgentFactory
from harbor.cli.notifications import show_registry_hint_if_first_run
from harbor.cli.utils import parse_env_vars, parse_kwargs
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import (
    JobConfig,
    LocalDatasetConfig,
    OrchestratorConfig,
    RegistryDatasetConfig,
)
from harbor.models.orchestrator_type import OrchestratorType
from harbor.models.registry import LocalRegistryInfo, RemoteRegistryInfo
from harbor.models.task.task import Task
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

jobs_app = Typer(no_args_is_help=True)
console = Console()

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


def print_job_results_tables(job_result) -> None:
    def _round_key_order(key: str) -> tuple[int, str]:
        match = re.fullmatch(r"round_(\d+)", key)
        if match:
            return (int(match.group(1)), key)
        return (10**9, key)

    def _reward_value_sort_key(value: float | int):
        try:
            return (-float(value), str(value))
        except Exception:
            return (0.0, str(value))

    def _reward_to_float(value: float | int) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    for evals_key, dataset_stats in job_result.stats.evals.items():
        # Parse the evals_key to extract agent_name, model_name (if present), and dataset_name
        # Key format is either "agent__model__dataset" or "agent__dataset"
        parts = evals_key.split("__")
        if len(parts) == 3:
            agent_name, model_name, dataset_name = parts
            display_agent = f"{agent_name} ({model_name})"
        else:
            agent_name, dataset_name = parts
            display_agent = agent_name

        # Create table
        table = Table(
            title=f"[bold]{display_agent}[/bold] on [bold]{dataset_name}[/bold]"
        )

        # Add columns
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add agent and dataset info
        table.add_row("Agent", display_agent)
        table.add_row("Dataset", dataset_name)
        table.add_row("Trials", str(dataset_stats.n_trials))
        table.add_row("Errors", str(dataset_stats.n_errors))

        # Add metrics
        if dataset_stats.metrics:
            table.add_row("", "")  # Separator
            for i, metric in enumerate(dataset_stats.metrics):
                for key, value in metric.items():
                    if isinstance(value, float):
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)
                    metric_label = (
                        f"Metric {i + 1}: {key}"
                        if len(dataset_stats.metrics) > 1
                        else key
                    )
                    table.add_row(metric_label.title(), formatted_value)

        # Add reward statistics (counts only, not the trial names)
        if dataset_stats.reward_stats:
            table.add_row("", "")  # Separator
            round_keys = sorted(
                [
                    reward_key
                    for reward_key in dataset_stats.reward_stats
                    if re.fullmatch(r"round_\d+", reward_key)
                ],
                key=_round_key_order,
            )
            has_multiround_rewards = len(round_keys) > 0

            if has_multiround_rewards:
                table.add_row("[bold]Multiround Reward Summary[/bold]", "")
            else:
                table.add_row("[bold]Reward Distribution[/bold]", "")

            ordered_reward_keys: list[str] = []
            ordered_reward_keys.extend(round_keys)
            if "reward" in dataset_stats.reward_stats:
                ordered_reward_keys.append("reward")
            ordered_reward_keys.extend(
                sorted(
                    reward_key
                    for reward_key in dataset_stats.reward_stats
                    if reward_key not in set(ordered_reward_keys)
                )
            )

            for reward_key in ordered_reward_keys:
                reward_values = dataset_stats.reward_stats[reward_key]
                label = reward_key
                if reward_key == "reward" and has_multiround_rewards:
                    label = "final_reward"

                if not has_multiround_rewards:
                    # Preserve single-round output behavior.
                    for reward_value, trial_names in reward_values.items():
                        count = len(trial_names)
                        table.add_row(f"  {label} = {reward_value}", str(count))
                    continue

                total_count = sum(len(trial_names) for trial_names in reward_values.values())
                numeric_entries: list[tuple[float, int]] = []
                for reward_value, trial_names in reward_values.items():
                    numeric_value = _reward_to_float(reward_value)
                    if numeric_value is None:
                        numeric_entries = []
                        break
                    numeric_entries.append((numeric_value, len(trial_names)))

                if total_count > 0 and numeric_entries:
                    mean_reward = (
                        sum(value * count for value, count in numeric_entries)
                        / total_count
                    )
                    table.add_row(
                        f"  {label} mean",
                        f"{mean_reward:.3f} (n={total_count})",
                    )

                    is_binary_reward = all(
                        value in (0.0, 1.0) for value, _ in numeric_entries
                    )
                    if is_binary_reward:
                        success_count = sum(
                            count for value, count in numeric_entries if value == 1.0
                        )
                        success_rate = success_count / total_count
                        table.add_row(
                            f"  {label} success",
                            f"{success_count}/{total_count} ({success_rate:.1%})",
                        )
                else:
                    bins = ", ".join(
                        f"{reward_value}:{len(trial_names)}"
                        for reward_value, trial_names in sorted(
                            reward_values.items(),
                            key=lambda item: _reward_value_sort_key(item[0]),
                        )
                    )
                    table.add_row(f"  {label} bins", bins if bins else "-")

        # Add exception statistics (counts only, not the trial names)
        if dataset_stats.exception_stats:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Exception Distribution[/bold]", "")
            for (
                exception_type,
                trial_names,
            ) in dataset_stats.exception_stats.items():
                count = len(trial_names)
                table.add_row(f"  {exception_type}", str(count))

        console.print(table)
        console.print()  # Add spacing between tables


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt


def _refresh_latest_snapshot_alias(trial_dir: Path, round_num: int) -> None:
    trial_paths = TrialPaths(trial_dir)
    round_snapshot_path = trial_paths.round_state_snapshot_path(round_num)
    round_archive_path = trial_paths.round_state_image_archive_path(round_num)
    latest_snapshot_path = trial_paths.state_snapshot_path
    latest_archive_path = trial_paths.state_image_archive_path

    if not round_snapshot_path.exists():
        return

    snapshot_payload = json.loads(round_snapshot_path.read_text())
    snapshot_payload["archive_path"] = str(latest_archive_path.expanduser().absolute())
    latest_snapshot_path.write_text(json.dumps(snapshot_payload, indent=2))

    if latest_archive_path.exists() or latest_archive_path.is_symlink():
        latest_archive_path.unlink()

    if not round_archive_path.exists():
        return

    try:
        latest_archive_path.symlink_to(
            round_archive_path.relative_to(latest_archive_path.parent)
        )
    except OSError:
        shutil.copy2(round_archive_path, latest_archive_path)


def _cleanup_trial_for_resume(trial_dir: Path, from_round: int) -> None:
    """Clean up an existing trial directory for in-place resume.

    Removes result.json, config.json, trial.log, verifier outputs for
    round >= from_round, and agent runtime artifacts. Keeps agent artifacts
    for round < from_round (e.g. round-1-command-*, claude-code.txt) and
    infrastructure files (install.sh, setup/).
    The caller must have already created a backup via copytree.
    """
    # Remove result.json so job resume picks it up
    result_path = trial_dir / "result.json"
    if result_path.exists():
        result_path.unlink()

    # Remove config.json so Job treats this as a fresh trial directory
    config_path = trial_dir / "config.json"
    if config_path.exists():
        config_path.unlink()

    # Remove trial.log from the previous run
    trial_log = trial_dir / "trial.log"
    if trial_log.exists():
        trial_log.unlink()

    exception_path = trial_dir / "exception.txt"
    if exception_path.exists():
        exception_path.unlink()

    # Remove verifier outputs for round >= from_round
    verifier_dir = trial_dir / "verifier"
    if verifier_dir.exists():
        for f in verifier_dir.iterdir():
            # round_N_*.txt / round_N_*.json files
            if f.name.startswith("round_"):
                parts = f.name.split("_", 2)  # round_N_rest
                if len(parts) >= 2:
                    try:
                        rn = int(parts[1])
                        if rn >= from_round:
                            f.unlink()
                    except ValueError:
                        pass

        # Remove aggregate files and last-round residual outputs
        for name in (
            "multiround_results.json",
            "reward.txt",
            "reward.json",
            "test-stdout.txt",
            "test-stderr.txt",
            "test_output.txt",
        ):
            p = verifier_dir / name
            if p.exists():
                p.unlink()

    # Clean agent directory: keep round < from_round artifacts, remove the rest.
    # Different agents use different naming:
    #   claude-code: round-N-command-*, claude-code-round-N.txt, sessions/
    #   terminus-2:  episode-* (overwritten each round, no round info in name),
    #                trajectory.json, recording.cast, terminus_2.pane
    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        for f in agent_dir.iterdir():
            name = f.name
            # claude-code: round-N-command-* directories
            if name.startswith("round-") and "-command-" in name:
                try:
                    rn = int(name.split("-")[1])
                    if rn >= from_round:
                        shutil.rmtree(f) if f.is_dir() else f.unlink()
                except (ValueError, IndexError):
                    pass
                continue
            # claude-code: claude-code-round-N.txt log files
            if name.startswith("claude-code-round-"):
                try:
                    rn = int(name.replace("claude-code-round-", "").split(".")[0])
                    if rn >= from_round:
                        f.unlink()
                except (ValueError, IndexError):
                    pass
                continue
            # oracle: round_N_oracle.txt / round_N_exit-code.txt
            if name.startswith("round_") and (
                name.endswith("_oracle.txt") or name.endswith("_exit-code.txt")
            ):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    try:
                        rn = int(parts[1])
                        if rn >= from_round:
                            f.unlink(missing_ok=True)
                    except ValueError:
                        pass
                continue
            # Claude session snapshots: preserve < from_round, drop future rounds.
            if name == "session_snapshots" and f.is_dir():
                for entry in f.iterdir():
                    if not entry.is_dir() or not entry.name.startswith("round_"):
                        if entry.is_dir():
                            shutil.rmtree(entry, ignore_errors=True)
                        else:
                            entry.unlink(missing_ok=True)
                        continue

                    try:
                        rn = int(entry.name.split("_", 1)[1])
                    except ValueError:
                        shutil.rmtree(entry, ignore_errors=True)
                        continue

                    if rn >= from_round:
                        shutil.rmtree(entry, ignore_errors=True)
                continue
            # Generic agent runtime snapshots: preserve < from_round, drop future rounds.
            if name == "runtime_snapshots" and f.is_dir():
                for entry in f.iterdir():
                    if not entry.is_dir() or not entry.name.startswith("round_"):
                        if entry.is_dir():
                            shutil.rmtree(entry, ignore_errors=True)
                        else:
                            entry.unlink(missing_ok=True)
                        continue

                    try:
                        rn = int(entry.name.split("_", 1)[1])
                    except ValueError:
                        shutil.rmtree(entry, ignore_errors=True)
                        continue

                    if rn >= from_round:
                        shutil.rmtree(entry, ignore_errors=True)
                continue
            # Keep: install.sh, setup/ (agent installation, not round-specific)
            if name in ("install.sh", "setup"):
                continue
            # Keep: claude-code.txt (round 1 main log) if from_round > 1
            if name == "claude-code.txt" and from_round > 1:
                continue
            # Remove everything else: episode-*, trajectory.json,
            # recording.cast, terminus_2.pane, sessions/, etc.
            # These are either overwritten each round or will be regenerated.
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)

    # Clean state directory: keep round-specific snapshots for rounds < from_round,
    # drop stale future-round snapshots, and repoint top-level latest to the last
    # preserved round so the resumed trial still has a coherent snapshot lineage.
    state_dir = trial_dir / "state"
    if state_dir.exists():
        preserved_rounds: list[int] = []

        latest_snapshot_path = state_dir / "snapshot.json"
        if latest_snapshot_path.exists():
            latest_snapshot_path.unlink()

        latest_archive_path = state_dir / "snapshot-image.tar"
        if latest_archive_path.exists() or latest_archive_path.is_symlink():
            latest_archive_path.unlink()

        for entry in state_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("round_"):
                continue

            try:
                round_num = int(entry.name.split("_", 1)[1])
            except ValueError:
                continue

            if round_num >= from_round:
                shutil.rmtree(entry, ignore_errors=True)
                continue

            if (entry / "snapshot.json").exists():
                preserved_rounds.append(round_num)

        if preserved_rounds:
            _refresh_latest_snapshot_alias(trial_dir, max(preserved_rounds))


def _looks_like_trial_output_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        TrialConfig.model_validate_json(config_path.read_text())
    except Exception:
        return False
    return True


def _resume_source_has_successful_round(
    resume_trial_dir: Path, *, start_round: int
) -> bool:
    result_path = resume_trial_dir / "result.json"
    if not result_path.exists():
        return False

    try:
        result_payload = json.loads(result_path.read_text())
    except Exception:
        return False

    if result_payload.get("exception_info") is not None:
        return False

    verifier_result = result_payload.get("verifier_result")
    if not isinstance(verifier_result, dict):
        return False
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict):
        return False

    round_reward = rewards.get(f"round_{start_round - 1}")
    if round_reward is None and start_round == 2:
        round_reward = rewards.get("reward")
    if round_reward is None:
        return False

    try:
        return float(round_reward) == 1.0
    except (TypeError, ValueError):
        return False


def _resume_source_has_snapshot_image(
    resume_trial_dir: Path, *, start_round: int
) -> bool:
    snapshot_meta_path = _resolve_resume_snapshot_metadata_path(
        resume_trial_dir,
        start_round=start_round,
    )
    if snapshot_meta_path is None:
        return False

    try:
        snapshot_meta = json.loads(snapshot_meta_path.read_text())
    except Exception:
        return False

    image_ref = snapshot_meta.get("image_tag") or snapshot_meta.get("image_ref")
    return isinstance(image_ref, str) and bool(image_ref.strip())


def _resume_source_matches_completed_round(
    resume_trial_dir: Path, *, start_round: int
) -> bool:
    completed_round = start_round - 1
    if completed_round < 1:
        return True

    latest_round = _resolve_resume_source_latest_round(resume_trial_dir)
    if latest_round is None:
        return True
    return latest_round == completed_round


def _resolve_resume_trial_dir(resume_path: Path, *, start_round: int) -> Path:
    resolved_path = resume_path.resolve()
    if not resolved_path.exists():
        raise ValueError(f"Resume trial directory does not exist: {resolved_path}")

    if _looks_like_trial_output_dir(resolved_path):
        return resolved_path

    if not resolved_path.is_dir():
        raise ValueError(f"Resume path is not a directory: {resolved_path}")

    candidate_dirs = sorted(
        (
            child
            for child in resolved_path.iterdir()
            if _looks_like_trial_output_dir(child)
            and _resume_source_matches_completed_round(
                child, start_round=start_round
            )
            and _resume_source_has_successful_round(child, start_round=start_round)
            and _resume_source_has_snapshot_image(child, start_round=start_round)
        ),
        key=lambda path: path.name,
    )
    if candidate_dirs:
        return candidate_dirs[0]

    raise ValueError(
        "--resume-trial did not resolve to a usable trial directory. "
        "If you pass a job/timestamp directory, it must contain at least one child "
        "trial whose latest completed round matches the requested resume source, "
        "whose previous round succeeded, and which has snapshot image metadata. "
        f"Path: {resolved_path}"
    )


def _load_resume_source_trial_config(resume_trial_dir: Path) -> TrialConfig:
    """Load the original trial config.json from a resume source directory."""
    config_path = resume_trial_dir / "config.json"
    if not config_path.exists():
        raise ValueError(
            f"--resume-trial requires an existing config.json in the source trial: {resume_trial_dir}"
        )

    try:
        return TrialConfig.model_validate_json(config_path.read_text())
    except Exception as e:
        raise ValueError(
            f"Failed to read source trial config from {config_path}: {e}"
        ) from e


def _load_resume_source_task_checksum(resume_trial_dir: Path) -> str | None:
    """Best-effort task checksum from the source trial's result.json."""
    result_path = resume_trial_dir / "result.json"
    if not result_path.exists():
        return None

    try:
        result_payload = json.loads(result_path.read_text())
    except Exception:
        return None

    task_checksum = result_payload.get("task_checksum")
    return task_checksum if isinstance(task_checksum, str) and task_checksum else None


def _is_volatile_task_artifact(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part.startswith("creating_"):
            return True
        if part in _VOLATILE_TASK_DIR_NAMES:
            return True

    if relative_path.name in _VOLATILE_TASK_FILE_NAMES:
        return True
    return False


def _compute_task_definition_checksum(task_dir: Path) -> str:
    digest = hashlib.sha256()

    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(task_dir)
        if _is_volatile_task_artifact(relative_path):
            continue

        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _resolve_resume_source_latest_round(resume_trial_dir: Path) -> int | None:
    trial_paths = TrialPaths(resume_trial_dir)
    latest_snapshot_path = trial_paths.state_snapshot_path
    if latest_snapshot_path.exists():
        try:
            latest_snapshot_payload = json.loads(latest_snapshot_path.read_text())
        except Exception:
            latest_snapshot_payload = {}

        latest_round = latest_snapshot_payload.get("round")
        if isinstance(latest_round, int):
            return latest_round

    round_results_path = trial_paths.verifier_dir / "multiround_results.json"
    if not round_results_path.exists():
        return None

    try:
        round_results = json.loads(round_results_path.read_text())
    except Exception:
        return None

    rounds = [
        round_num
        for item in round_results
        if isinstance(item, dict)
        for round_num in [item.get("round")]
        if isinstance(round_num, int)
    ]
    return max(rounds) if rounds else None


def _resolve_resume_snapshot_metadata_path(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    trial_paths = TrialPaths(resume_trial_dir)
    completed_round = start_round - 1

    if completed_round >= 1:
        round_snapshot_path = trial_paths.round_state_snapshot_path(completed_round)
        if round_snapshot_path.exists():
            return round_snapshot_path

        latest_snapshot_path = trial_paths.state_snapshot_path
        if latest_snapshot_path.exists():
            try:
                latest_snapshot_payload = json.loads(latest_snapshot_path.read_text())
            except Exception:
                latest_snapshot_payload = {}

            latest_round = latest_snapshot_payload.get("round")
            if latest_round in (None, completed_round):
                return latest_snapshot_path
        return None

    latest_snapshot_path = trial_paths.state_snapshot_path
    if latest_snapshot_path.exists():
        return latest_snapshot_path
    return None


def _resolve_resume_claude_sessions_dir(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    trial_paths = TrialPaths(resume_trial_dir)
    completed_round = start_round - 1
    if completed_round < 1:
        return None

    round_sessions_dir = trial_paths.agent_round_sessions_dir(completed_round)
    if round_sessions_dir.exists():
        return round_sessions_dir

    latest_sessions_dir = trial_paths.agent_sessions_dir
    if not latest_sessions_dir.exists():
        return None

    latest_round = _resolve_resume_source_latest_round(resume_trial_dir)
    if latest_round != completed_round:
        return None
    return latest_sessions_dir


def _resolve_resume_terminus_runtime_state_path(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    trial_paths = TrialPaths(resume_trial_dir)
    completed_round = start_round - 1
    if completed_round < 1:
        return None

    round_state_path = trial_paths.terminus_round_runtime_state_path(completed_round)
    if round_state_path.exists():
        return round_state_path
    return None


def _validate_resume_task_identity(
    target_task: Task,
    target_task_config: TaskConfig,
    resume_trial_dir: Path,
) -> None:
    """Ensure the resume source trial was produced by the same task definition."""
    source_trial_config = _load_resume_source_trial_config(resume_trial_dir)

    source_task_path = source_trial_config.task.path
    if source_task_path.exists():
        source_checksum = _compute_task_definition_checksum(source_task_path)
        target_checksum = _compute_task_definition_checksum(target_task.task_dir)
        if source_checksum != target_checksum:
            raise ValueError(
                "--resume-trial source task does not match the current task. "
                "Resume requires the same task definition."
            )
        return

    source_checksum = _load_resume_source_task_checksum(resume_trial_dir)
    if source_checksum is None:
        try:
            source_checksum = Task(source_trial_config.task.path).checksum
        except Exception:
            source_checksum = None

    if source_checksum is not None:
        if source_checksum != target_task.checksum:
            raise ValueError(
                "--resume-trial source task does not match the current task. "
                "Resume requires the same task definition."
            )
        return

    if source_trial_config.task.get_task_id() != target_task_config.get_task_id():
        raise ValueError(
            "--resume-trial source task does not match the current task. "
        "Resume requires the same task definition."
        )


def _preflight_multiround_resume_source(
    *,
    resume_trial_dir: Path,
    start_round: int,
    target_agent_config: AgentConfig,
    policy: str,
) -> None:
    if policy == "off":
        return

    snapshot_meta_path = _resolve_resume_snapshot_metadata_path(
        resume_trial_dir,
        start_round=start_round,
    )
    if snapshot_meta_path is None:
        raise ValueError(
            "--resume-trial preflight failed: no matching resume snapshot metadata "
            f"was found for round {start_round - 1} in {resume_trial_dir}"
        )

    try:
        snapshot_payload = json.loads(snapshot_meta_path.read_text())
    except Exception as e:
        raise ValueError(
            "--resume-trial preflight failed: could not read resume snapshot metadata "
            f"from {snapshot_meta_path}: {e}"
        ) from e

    if not (snapshot_payload.get("image_tag") or snapshot_payload.get("image_ref")):
        raise ValueError(
            "--resume-trial preflight failed: resume snapshot metadata does not "
            f"contain an image reference: {snapshot_meta_path}"
        )

    if policy == "snapshot":
        return

    source_trial_config = _load_resume_source_trial_config(resume_trial_dir)
    source_agent_name = AgentFactory.get_agent_class_from_config(
        source_trial_config.agent
    ).name()
    target_agent_name = AgentFactory.get_agent_class_from_config(target_agent_config).name()

    if (
        source_agent_name == AgentName.ORACLE.value
        and target_agent_name != AgentName.ORACLE.value
    ):
        return

    completed_round = start_round - 1
    if target_agent_name == AgentName.CLAUDE_CODE.value:
        if (
            _resolve_resume_claude_sessions_dir(
                resume_trial_dir,
                start_round=start_round,
            )
            is None
        ):
            raise ValueError(
                "--resume-trial preflight failed: no Claude session snapshot was "
                f"found for round {completed_round} in {resume_trial_dir}"
            )
    elif target_agent_name == AgentName.TERMINUS_2.value:
        if (
            _resolve_resume_terminus_runtime_state_path(
                resume_trial_dir,
                start_round=start_round,
            )
            is None
        ):
            raise ValueError(
                "--resume-trial preflight failed: no Terminus-2 runtime snapshot was "
                f"found for round {completed_round} in {resume_trial_dir}"
            )


def _validate_resume_agent_transition(
    target_agent_config: AgentConfig, resume_trial_dir: Path
) -> None:
    """Validate that the requested resume source/target agent transition is allowed."""
    source_trial_config = _load_resume_source_trial_config(resume_trial_dir)

    target_agent_name = AgentFactory.get_agent_class_from_config(target_agent_config).name()
    source_agent_name = AgentFactory.get_agent_class_from_config(
        source_trial_config.agent
    ).name()

    if source_agent_name == target_agent_name:
        return

    if source_agent_name == AgentName.ORACLE.value:
        return

    if target_agent_name == AgentName.ORACLE.value:
        raise ValueError(
            "--resume-trial does not support resuming into oracle from a non-oracle "
            "source. Oracle can prepare later rounds, but it does not consume another "
            "agent's continuation state. Use --start-round N for the default "
            "Oracle-prepared late-round start."
        )

    raise ValueError(
        "--resume-trial only supports same-agent continuation, plus oracle -> "
        "target-agent handoff. "
        f"Got source agent '{source_agent_name}' and target agent "
        f"'{target_agent_name}'."
    )


def _validate_resume_shape(config: JobConfig) -> None:
    if len(config.agents) != 1:
        raise ValueError("--resume-trial requires exactly one agent")


def _would_enable_roundwise_multiround_attempt_selection(config: JobConfig) -> bool:
    if config.n_attempts <= 1:
        return False
    if (config.jobs_dir / config.job_name / "result.json").exists():
        return False

    task_configs = list(config.tasks)
    for dataset in config.datasets:
        task_configs.extend(
            dataset.get_task_configs(disable_verification=config.verifier.disable)
        )

    if not task_configs:
        return False

    for task_config in task_configs:
        if task_config.is_git_task():
            return False
        if not Task(task_config.path).is_multiround:
            return False

    return True


def start(
    config_path: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="A job configuration path in yaml or json format. "
            "Should implement the schema of harbor.models.job.config:JobConfig. "
            "Allows for more granular control over the job configuration.",
            rich_help_panel="Config",
            show_default=False,
        ),
    ] = None,
    job_name: Annotated[
        str | None,
        Option(
            "--job-name",
            help="Name of the job (default: timestamp)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    jobs_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--jobs-dir",
            help=(
                "Directory to store job results (default: "
                f"{JobConfig.model_fields['jobs_dir'].default})"
            ),
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_attempts: Annotated[
        int | None,
        Option(
            "-k",
            "--n-attempts",
            help=(
                "Number of attempts per trial (default: "
                f"{JobConfig.model_fields['n_attempts'].default})"
            ),
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        Option(
            "--timeout-multiplier",
            help=(
                "Multiplier for task timeouts (default: "
                f"{JobConfig.model_fields['timeout_multiplier'].default})"
            ),
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-timeout-multiplier",
            help="Multiplier for agent execution timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    verifier_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--verifier-timeout-multiplier",
            help="Multiplier for verifier timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    agent_setup_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--agent-setup-timeout-multiplier",
            help="Multiplier for agent setup timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    environment_build_timeout_multiplier: Annotated[
        float | None,
        Option(
            "--environment-build-timeout-multiplier",
            help="Multiplier for environment build timeout (overrides --timeout-multiplier)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[
        bool,
        Option(
            "-q",
            "--quiet",
            "--silent",
            help="Suppress individual trial progress displays",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    debug: Annotated[
        bool,
        Option(
            "--debug",
            help="Enable debug logging",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    orchestrator_type: Annotated[
        OrchestratorType | None,
        Option(
            "--orchestrator",
            help=(
                "Orchestrator type (default: "
                f"{OrchestratorConfig.model_fields['type'].default.value})"
            ),
            rich_help_panel="Orchestrator",
            show_default=False,
        ),
    ] = None,
    n_concurrent_trials: Annotated[
        int | None,
        Option(
            "-n",
            "--n-concurrent",
            help=(
                "Number of concurrent trials to run (default: "
                f"{OrchestratorConfig.model_fields['n_concurrent_trials'].default})"
            ),
            rich_help_panel="Orchestrator",
            show_default=False,
        ),
    ] = None,
    orchestrator_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ok",
            "--orchestrator-kwarg",
            help="Orchestrator kwarg in key=value format (can be used multiple times)",
            rich_help_panel="Orchestrator",
            show_default=False,
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        Option(
            "-r",
            "--max-retries",
            help="Maximum number of retry attempts (default: 0)",
            rich_help_panel="Orchestrator",
            show_default=False,
        ),
    ] = None,
    retry_include_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-include",
            help="Exception types to retry on. If not specified, all exceptions except "
            "those in --retry-exclude are retried (can be used multiple times)",
            rich_help_panel="Orchestrator",
            show_default=False,
        ),
    ] = None,
    retry_exclude_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-exclude",
            help="Exception types to NOT retry on (can be used multiple times)",
            rich_help_panel="Orchestrator",
        ),
    ] = [
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    ],
    agent_name: Annotated[
        AgentName | None,
        Option(
            "-a",
            "--agent",
            help=f"Agent name (default: {AgentName.ORACLE.value})",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_import_path: Annotated[
        str | None,
        Option(
            "--agent-import-path",
            help="Import path for custom agent",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    model_names: Annotated[
        list[str] | None,
        Option(
            "-m",
            "--model",
            help="Model name for the agent (can be used multiple times)",
            rich_help_panel="Agent",
            show_default=True,
        ),
    ] = None,
    agent_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ak",
            "--agent-kwarg",
            help="Additional agent kwarg in the format 'key=value'. You can view "
            "available kwargs by looking at the agent's `__init__` method. "
            "Can be set multiple times to set multiple kwargs. Common kwargs "
            "include: version, prompt_template, etc.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    agent_env: Annotated[
        list[str] | None,
        Option(
            "--ae",
            "--agent-env",
            help="Environment variable to pass to the agent in KEY=VALUE format. "
            "Can be used multiple times. Example: --ae AWS_REGION=us-east-1",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    environment_type: Annotated[
        EnvironmentType | None,
        Option(
            "-e",
            "--env",
            help=f"Environment type (default: {EnvironmentType.DOCKER.value})",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_import_path: Annotated[
        str | None,
        Option(
            "--environment-import-path",
            help="Import path for custom environment (module.path:ClassName).",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_force_build: Annotated[
        bool | None,
        Option(
            "--force-build/--no-force-build",
            help=(
                "Whether to force rebuild the environment (default: "
                f"{'--force-build' if EnvironmentConfig.model_fields['force_build'].default else '--no-force-build'})"
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_delete: Annotated[
        bool | None,
        Option(
            "--delete/--no-delete",
            help=(
                "Whether to delete the environment after completion (default: "
                f"{'--delete' if EnvironmentConfig.model_fields['delete'].default else '--no-delete'})"
            ),
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_cpus: Annotated[
        int | None,
        Option(
            "--override-cpus",
            help="Override the number of CPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_memory_mb: Annotated[
        int | None,
        Option(
            "--override-memory-mb",
            help="Override the memory (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_storage_mb: Annotated[
        int | None,
        Option(
            "--override-storage-mb",
            help="Override the storage (in MB) for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    override_gpus: Annotated[
        int | None,
        Option(
            "--override-gpus",
            help="Override the number of GPUs for the environment",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_kwargs: Annotated[
        list[str] | None,
        Option(
            "--ek",
            "--environment-kwarg",
            help="Environment kwarg in key=value format (can be used multiple times)",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    path: Annotated[
        Path | None,
        Option(
            "-p",
            "--path",
            help="Path to a local task or dataset directory",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    task_git_url: Annotated[
        str | None,
        Option(
            help="Git URL for a task repository",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    task_git_commit_id: Annotated[
        str | None,
        Option(
            "--task-git-commit",
            help="Git commit ID for the task (requires --task-git-url)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_name_version: Annotated[
        str | None,
        Option(
            "-d",
            "--dataset",
            help="Dataset name@version (e.g., 'dataset@1.0')",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option(
            "--registry-url",
            help="Registry URL for remote dataset",
            rich_help_panel="Dataset",
            show_default="The default harbor registry.",
        ),
    ] = None,
    registry_path: Annotated[
        Path | None,
        Option(
            "--registry-path",
            help="Path to local registry for dataset",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_task_names: Annotated[
        list[str] | None,
        Option(
            "-t",
            "--task-name",
            help="Task name to include from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_exclude_task_names: Annotated[
        list[str] | None,
        Option(
            "-x",
            "--exclude-task-name",
            help="Task name to exclude from dataset (supports glob patterns)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option(
            "-l",
            "--n-tasks",
            help="Maximum number of tasks to run (applied after other filters)",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    export_traces: Annotated[
        bool,
        Option(
            "--export-traces/--no-export-traces",
            help="After job completes, export traces from the job directory",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = False,
    export_sharegpt: Annotated[
        bool,
        Option(
            "--export-sharegpt/--no-export-sharegpt",
            help="Also emit ShareGPT column when exporting traces",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = False,
    export_episodes: Annotated[
        str,
        Option(
            "--export-episodes",
            help="Which episodes to export per trial (all|last)",
            rich_help_panel="Traces",
        ),
    ] = "all",
    export_push: Annotated[
        bool,
        Option(
            "--export-push/--no-export-push",
            help="Push exported dataset to HF Hub",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = False,
    export_repo: Annotated[
        str | None,
        Option(
            "--export-repo",
            help="Target HF repo id (org/name) when pushing traces",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = None,
    export_instruction_metadata: Annotated[
        bool,
        Option(
            "--export-instruction-metadata/--no-export-instruction-metadata",
            help="Include instruction text column when exporting traces",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = False,
    export_verifier_metadata: Annotated[
        bool,
        Option(
            "--export-verifier-metadata/--no-export-verifier-metadata",
            help="Include verifier stdout/stderr column when exporting traces",
            rich_help_panel="Traces",
            show_default=False,
        ),
    ] = False,
    artifact_paths: Annotated[
        list[str] | None,
        Option(
            "--artifact",
            help="Environment path to download as an artifact after the trial "
            "(can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    disable_verification: Annotated[
        bool,
        Option(
            "--disable-verification/--enable-verification",
            help="Disable task verification (skip running tests)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = False,
    multiround_continue_successes_per_round: Annotated[
        int | None,
        Option(
            "--multiround-continue-successes-per-round",
            help="In multi-round mode with --n-attempts > 1, number of successful "
            "trajectories selected per round to continue to the next round "
            "(default: 1, picks the first successful trajectory)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_state_cache_policy: Annotated[
        str | None,
        Option(
            "--multiround-state-cache-policy",
            help="Container state snapshot caching policy for multi-round runs: "
            "off, success, all (default: success)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_resume_preflight_policy: Annotated[
        str | None,
        Option(
            "--multiround-resume-preflight-policy",
            help="Resume preflight policy for multi-round resume: "
            "off, snapshot, strict (default: strict)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    max_round: Annotated[
        int | None,
        Option(
            "--max-round",
            help="Maximum round number to execute (e.g. --max-round 3 runs rounds 1-3)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_aggregate_start_round: Annotated[
        int | None,
        Option(
            "--multiround-aggregate-start-round",
            help="Override the first round included in final multi-round reward aggregation",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_aggregate_end_round: Annotated[
        int | None,
        Option(
            "--multiround-aggregate-end-round",
            help="Override the last round included in final multi-round reward aggregation",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    start_round: Annotated[
        int | None,
        Option(
            "--start-round",
            help="Start from round N using the default Oracle-prepared late-round "
            "behavior, fast-forwarding earlier rounds with oracle solution "
            "(e.g. --start-round 3 fast-forwards rounds 1-2)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    resume_trial: Annotated[
        Path | None,
        Option(
            "--resume-trial",
            help="Path to an existing trial directory to resume from "
            "(requires --resume-round)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    resume_round: Annotated[
        int | None,
        Option(
            "--resume-round",
            help="Round number to resume from (requires --resume-trial)",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    no_resume_backup: Annotated[
        bool,
        Option(
            "--no-resume-backup",
            help="Skip creating __resumed_<timestamp> backup directories when "
            "using --resume-trial. The source trial is cleaned in-place and "
            "the resume source points to itself. Saves disk space at the cost "
            "of losing the pre-resume trial snapshot.",
            rich_help_panel="Multi-round",
            show_default=True,
        ),
    ] = False,
    output_jobs_dir: Annotated[
        Path | None,
        Option(
            "--output-jobs-dir",
            help="Write resume output to this jobs directory instead of the "
            "source trial's directory. The source trial is left unmodified "
            "(read-only). Only valid with --resume-trial.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
):
    from harbor.job import Job

    base_config = None
    if config_path is not None:
        if config_path.suffix == ".yaml":
            base_config = JobConfig.model_validate(
                yaml.safe_load(config_path.read_text())
            )
        elif config_path.suffix == ".json":
            base_config = JobConfig.model_validate_json(config_path.read_text())
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    config = base_config if base_config is not None else JobConfig()

    if job_name is not None:
        config.job_name = job_name
    if jobs_dir is not None:
        config.jobs_dir = jobs_dir
    if n_attempts is not None:
        config.n_attempts = n_attempts
    if timeout_multiplier is not None:
        config.timeout_multiplier = timeout_multiplier
    if agent_timeout_multiplier is not None:
        config.agent_timeout_multiplier = agent_timeout_multiplier
    if verifier_timeout_multiplier is not None:
        config.verifier_timeout_multiplier = verifier_timeout_multiplier
    if agent_setup_timeout_multiplier is not None:
        config.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
    if environment_build_timeout_multiplier is not None:
        config.environment_build_timeout_multiplier = (
            environment_build_timeout_multiplier
        )
    if debug:
        config.debug = debug

    if orchestrator_type is not None:
        config.orchestrator.type = orchestrator_type
    if n_concurrent_trials is not None:
        config.orchestrator.n_concurrent_trials = n_concurrent_trials
    if quiet:
        config.orchestrator.quiet = quiet
    if max_retries is not None:
        config.orchestrator.retry.max_retries = max_retries
    if retry_include_exceptions is not None:
        config.orchestrator.retry.include_exceptions = set(retry_include_exceptions)
    if retry_exclude_exceptions is not None:
        config.orchestrator.retry.exclude_exceptions = set(retry_exclude_exceptions)
    if orchestrator_kwargs is not None:
        config.orchestrator.kwargs.update(parse_kwargs(orchestrator_kwargs))

    if agent_name is not None or agent_import_path is not None:
        config.agents = []
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)

        if model_names is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                )
            ]

    if environment_type is not None:
        config.environment.type = environment_type
    if environment_import_path is not None:
        config.environment.import_path = environment_import_path
        config.environment.type = None  # Clear type so import_path takes precedence
    if environment_force_build is not None:
        config.environment.force_build = environment_force_build
    if environment_delete is not None:
        config.environment.delete = environment_delete
    if override_cpus is not None:
        config.environment.override_cpus = override_cpus
    if override_memory_mb is not None:
        config.environment.override_memory_mb = override_memory_mb
    if override_storage_mb is not None:
        config.environment.override_storage_mb = override_storage_mb
    if override_gpus is not None:
        config.environment.override_gpus = override_gpus
    if environment_kwargs is not None:
        config.environment.kwargs.update(parse_kwargs(environment_kwargs))

    if disable_verification:
        config.verifier.disable = disable_verification

    if multiround_continue_successes_per_round is not None:
        if multiround_continue_successes_per_round < 1:
            raise ValueError(
                "--multiround-continue-successes-per-round must be >= 1"
            )
        config.verifier.multiround_continue_successes_per_round = (
            multiround_continue_successes_per_round
        )

    if multiround_state_cache_policy is not None:
        allowed_policies = {"off", "success", "all"}
        if multiround_state_cache_policy not in allowed_policies:
            raise ValueError(
                "--multiround-state-cache-policy must be one of: off, success, all"
            )
        config.verifier.multiround_state_cache_policy = multiround_state_cache_policy

    if multiround_resume_preflight_policy is not None:
        allowed_preflight_policies = {"off", "snapshot", "strict"}
        if multiround_resume_preflight_policy not in allowed_preflight_policies:
            raise ValueError(
                "--multiround-resume-preflight-policy must be one of: "
                "off, snapshot, strict"
            )
        config.verifier.multiround_resume_preflight_policy = (
            multiround_resume_preflight_policy
        )

    pending_resume: tuple[Path, int] | None = None

    # --- Multi-round resume/start/max validation and wiring ---
    if resume_round is not None and resume_trial is None:
        raise ValueError("--resume-round requires --resume-trial")

    if resume_trial is not None:
        if resume_round is None:
            raise ValueError("--resume-trial requires --resume-round")
        if resume_round < 2:
            raise ValueError("--resume-round must be >= 2")
        if start_round is not None:
            raise ValueError(
                "--start-round and --resume-trial are mutually exclusive "
                "(--resume-trial sets start_round automatically)"
            )

        resolved_resume = _resolve_resume_trial_dir(
            resume_trial,
            start_round=resume_round,
        )
        pending_resume = (resolved_resume, resume_round)

    if start_round is not None and resume_trial is None:
        if start_round < 1:
            raise ValueError("--start-round must be >= 1")
        config.verifier.multiround_start_round = start_round

    if max_round is not None:
        if max_round < 1:
            raise ValueError("--max-round must be >= 1")
        config.verifier.multiround_max_round = max_round

    if multiround_aggregate_start_round is not None:
        if multiround_aggregate_start_round < 1:
            raise ValueError("--multiround-aggregate-start-round must be >= 1")
        config.verifier.multiround_aggregate_start_round = (
            multiround_aggregate_start_round
        )

    if multiround_aggregate_end_round is not None:
        if multiround_aggregate_end_round < 1:
            raise ValueError("--multiround-aggregate-end-round must be >= 1")
        config.verifier.multiround_aggregate_end_round = (
            multiround_aggregate_end_round
        )

    effective_start = config.verifier.multiround_start_round or 1
    if config.verifier.multiround_max_round is not None:
        if effective_start > config.verifier.multiround_max_round:
            raise ValueError(
                f"--start-round ({effective_start}) must be <= "
                f"--max-round ({config.verifier.multiround_max_round})"
            )

    if artifact_paths is not None:
        config.artifacts = list(artifact_paths)

    task_specified = task_git_url is not None or task_git_commit_id is not None

    dataset_specified = (
        dataset_name_version is not None
        or registry_url is not None
        or registry_path is not None
        or dataset_task_names is not None
        or dataset_exclude_task_names is not None
        or n_tasks is not None
    )

    if task_specified and dataset_specified:
        raise ValueError("Cannot specify both task and dataset parameters")

    if path is not None:
        task_paths = TaskPaths(path)
        is_task = task_paths.is_valid(disable_verification=disable_verification)

        if is_task:
            config.tasks = [
                TaskConfig(
                    path=path,
                    git_url=task_git_url,
                    git_commit_id=task_git_commit_id,
                )
            ]
            config.datasets = []
        else:
            config.tasks = []
            config.datasets = [
                LocalDatasetConfig(
                    path=path,
                    task_names=dataset_task_names,
                    exclude_task_names=dataset_exclude_task_names,
                    n_tasks=n_tasks,
                )
            ]

    elif task_specified:
        raise ValueError("Task configuration with --task-git-url requires --path")

    elif dataset_specified:
        config.tasks = []

        if dataset_name_version is not None:
            if "@" in dataset_name_version:
                name, version = dataset_name_version.split("@", 1)
            else:
                name, version = dataset_name_version, None

            if registry_url is not None:
                registry_info = RemoteRegistryInfo(url=registry_url)
            elif registry_path is not None:
                registry_info = LocalRegistryInfo(path=registry_path)
            else:
                registry_info = RemoteRegistryInfo()

            config.datasets = [
                RegistryDatasetConfig(
                    registry=registry_info,
                    name=name,
                    version=version,
                    task_names=dataset_task_names,
                    exclude_task_names=dataset_exclude_task_names,
                    n_tasks=n_tasks,
                )
            ]

        else:
            raise ValueError(
                "Cannot specify --registry-url, --registry-path, --task-name, or "
                "--exclude-task-name without also specifying --dataset or --path."
            )

    if config.verifier.multiround_start_round is not None:
        start_round_value = config.verifier.multiround_start_round
        task_configs = list(config.tasks)
        for dataset in config.datasets:
            task_configs.extend(
                dataset.get_task_configs(disable_verification=config.verifier.disable)
            )

        for task_config in task_configs:
            if task_config.is_git_task():
                continue

            task = Task(task_config.path)
            if start_round_value > task.num_rounds:
                raise ValueError(
                    f"--start-round ({start_round_value}) must be <= "
                    f"the task's num_rounds ({task.num_rounds}) for task '{task.name}'"
                )

    task_configs = list(config.tasks)
    for dataset in config.datasets:
        task_configs.extend(
            dataset.get_task_configs(disable_verification=config.verifier.disable)
        )

    for task_config in task_configs:
        if task_config.is_git_task():
            continue

        task = Task(task_config.path)
        if task.is_multiround and config.verifier.disable:
            raise ValueError(
                "--disable-verification is not supported for multi-round tasks. "
                "Multi-round reward aggregation and fanout selection require "
                "per-round verification."
            )

        aggregate_start = config.verifier.multiround_aggregate_start_round
        aggregate_end = config.verifier.multiround_aggregate_end_round
        if aggregate_start is not None and aggregate_start > task.num_rounds:
            raise ValueError(
                f"--multiround-aggregate-start-round ({aggregate_start}) must be <= "
                f"the task's num_rounds ({task.num_rounds}) for task '{task.name}'"
            )
        if aggregate_end is not None and aggregate_end > task.num_rounds:
            raise ValueError(
                f"--multiround-aggregate-end-round ({aggregate_end}) must be <= "
                f"the task's num_rounds ({task.num_rounds}) for task '{task.name}'"
            )
        if (
            aggregate_start is not None
            and aggregate_end is not None
            and aggregate_start > aggregate_end
        ):
            raise ValueError(
                f"--multiround-aggregate-start-round ({aggregate_start}) must be <= "
                f"--multiround-aggregate-end-round ({aggregate_end})"
            )

    if output_jobs_dir is not None and pending_resume is None:
        raise ValueError(
            "--output-jobs-dir requires --resume-trial"
        )

    if pending_resume is not None:
        resolved_resume, resolved_resume_round = pending_resume

        _validate_resume_shape(config)

        if config.datasets:
            raise ValueError(
                "--resume-trial is not supported when running a dataset or multiple "
                "tasks; resume requires a single task job"
            )

        if len(config.tasks) != 1:
            raise ValueError(
                "--resume-trial requires exactly one task in the job configuration"
            )

        task = Task(config.tasks[0].path)
        _validate_resume_task_identity(task, config.tasks[0], resolved_resume)
        if resolved_resume_round > task.num_rounds:
            raise ValueError(
                f"--resume-round ({resolved_resume_round}) must be <= "
                f"the task's num_rounds ({task.num_rounds})"
            )

        if (
            config.verifier.multiround_max_round is not None
            and resolved_resume_round > config.verifier.multiround_max_round
        ):
            raise ValueError(
                f"--resume-round ({resolved_resume_round}) must be <= "
                f"--max-round ({config.verifier.multiround_max_round})"
            )

        for agent_config in config.agents:
            _validate_resume_agent_transition(agent_config, resolved_resume)
            _preflight_multiround_resume_source(
                resume_trial_dir=resolved_resume,
                start_round=resolved_resume_round,
                target_agent_config=agent_config,
                policy=config.verifier.multiround_resume_preflight_policy,
            )

        # Infer jobs_dir and job_name from the trial path:
        # trial_dir = JOBS_DIR / JOB_NAME / TRIAL_NAME
        trial_parent = resolved_resume.parent
        inferred_jobs_dir = trial_parent.parent
        inferred_job_name = trial_parent.name
        resume_trial_name = resolved_resume.name

        if output_jobs_dir is not None:
            # --output-jobs-dir: write to a separate directory, leave source
            # trial completely untouched (read-only).
            config.jobs_dir = output_jobs_dir
            # job_name will be auto-generated by Job if not explicitly set
            resume_source_dir = resolved_resume
            # Not an in-place resume: set resume_trial_name to None so
            # fast-forwarded rounds are marked as "fast_forwarded" rather
            # than preserving the source trial's status labels; but still
            # set resume_source so runtime can read snapshots/results.
            resume_trial_name_for_config = None
        else:
            # Legacy in-place resume: output goes back to source trial dir.
            if jobs_dir is not None and jobs_dir.resolve() != inferred_jobs_dir.resolve():
                raise ValueError(
                    f"-o ({jobs_dir.resolve()}) conflicts with --resume-trial path "
                    f"(inferred jobs_dir: {inferred_jobs_dir})"
                )

            config.jobs_dir = inferred_jobs_dir
            config.job_name = inferred_job_name
            resume_trial_name_for_config = resume_trial_name

            if no_resume_backup:
                # Save multiround_results.json before cleanup since
                # _copy_resume_round_results needs it at runtime.
                verifier_dir = resolved_resume / "verifier"
                saved_multiround_results = None
                mr_path = verifier_dir / "multiround_results.json"
                if mr_path.exists():
                    saved_multiround_results = mr_path.read_bytes()

                _cleanup_trial_for_resume(resolved_resume, resolved_resume_round)

                if saved_multiround_results is not None:
                    mr_path.parent.mkdir(parents=True, exist_ok=True)
                    mr_path.write_bytes(saved_multiround_results)

                resume_source_dir = resolved_resume
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = trial_parent / f"{resume_trial_name}__resumed_{ts}"
                shutil.copytree(resolved_resume, backup_dir, symlinks=True)

                _cleanup_trial_for_resume(resolved_resume, resolved_resume_round)

                resume_source_dir = backup_dir

            for name in ("result.json", "config.json"):
                p = trial_parent / name
                if p.exists():
                    p.unlink()

        config.verifier.multiround_start_round = resolved_resume_round
        config.verifier.multiround_resume_source = str(resume_source_dir)
        config.verifier.multiround_resume_trial_name = resume_trial_name_for_config

    if (
        config.verifier.multiround_state_cache_policy == "off"
        and _would_enable_roundwise_multiround_attempt_selection(config)
    ):
        raise ValueError(
            "-k/--n-attempts > 1 with --multiround-state-cache-policy off "
            "is only unsupported when round-wise multi-round attempt selection "
            "would be enabled: fanout child trials require parent snapshots "
            "which are never saved under policy 'off'. "
            "Use 'success' (default) or 'all' instead."
        )

    job = Job(config)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    show_registry_hint_if_first_run(console)

    job_result = asyncio.run(job.run())

    # Print results tables
    print_job_results_tables(job_result)

    if export_traces:
        from harbor.utils.traces_utils import export_traces as _export_traces

        job_dir = job.job_dir
        print(f"[traces] Exporting traces from job dir: {job_dir}")
        try:
            if export_push and not export_repo:
                raise ValueError("--export-push requires --export-repo <org/name>")
            ds = _export_traces(
                root=job_dir,
                recursive=True,
                episodes="last" if export_episodes == "last" else "all",
                to_sharegpt=export_sharegpt,
                repo_id=export_repo,
                push=export_push,
                include_instruction=export_instruction_metadata,
                include_verifier_output=export_verifier_metadata,
            )
            print(f"[traces] Exported {len(ds)} rows from {job_dir}")
        except Exception as e:
            print(f"[traces] Export failed: {e}")


@jobs_app.command()
def resume(
    job_path: Annotated[
        Path,
        Option(
            "-p",
            "--job-path",
            help="Path to the job directory containing the config.json file",
        ),
    ],
    filter_error_types: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter-error-type",
            help="Remove trials with these error types before resuming (can be used "
            "multiple times)",
            show_default=False,
        ),
    ] = ["CancelledError"],
):
    """Resume an existing job from its job directory."""
    from harbor.job import Job

    job_dir = Path(job_path)
    config_path = job_dir / "config.json"

    if not job_dir.exists():
        raise ValueError(f"Job directory does not exist: {job_dir}")

    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    if filter_error_types:
        filter_error_types_set = set(filter_error_types)
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue

            trial_paths = TrialPaths(trial_dir)

            if not trial_paths.result_path.exists():
                continue

            trial_result = TrialResult.model_validate_json(
                trial_paths.result_path.read_text()
            )
            if (
                trial_result.exception_info is not None
                and trial_result.exception_info.exception_type in filter_error_types_set
            ):
                console.print(
                    "Removing trial directory with "
                    f"{trial_result.exception_info.exception_type}: {trial_dir.name}"
                )
                shutil.rmtree(trial_dir)

    config = JobConfig.model_validate_json(config_path.read_text())
    job = Job(config)
    job_result = asyncio.run(job.run())

    # Print results tables
    print_job_results_tables(job_result)


@jobs_app.command()
def summarize(
    job_path: Annotated[
        Path,
        Argument(
            help="Path to a job directory or a parent directory containing multiple job subdirectories",
        ),
    ],
    n_concurrent: Annotated[
        int,
        Option(
            "-n",
            "--n-concurrent",
            help="Maximum number of concurrent summarization queries",
        ),
    ] = 5,
    model: Annotated[
        str | None,
        Option(
            "-m",
            "--model",
            help="Model to use for summarization (e.g., 'haiku', 'sonnet', 'opus')",
        ),
    ] = "haiku",
    all_trials: Annotated[
        bool,
        Option(
            "--all/--failed",
            help="Analyze all trials (by default, only failed trials are analyzed)",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        Option(
            "--overwrite",
            help="Overwrite existing summary.md files (by default, skip trials with existing summaries)",
        ),
    ] = False,
):
    """Summarize trial failures in a job using Claude Agent SDK."""
    from harbor.cli.summarize.summarizer import Summarizer

    if not job_path.exists():
        console.print(f"[red]Error: Job directory does not exist: {job_path}[/red]")
        raise SystemExit(1)

    if not job_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {job_path}[/red]")
        raise SystemExit(1)

    # Single job: config.json exists directly in job_path
    is_multi_job = not (job_path / "config.json").exists()
    if not is_multi_job:
        job_dirs = [job_path]
    else:
        # Multi-job: find all subdirectories that contain config.json
        job_dirs = sorted(
            d for d in job_path.iterdir() if d.is_dir() and (d / "config.json").exists()
        )
        if not job_dirs:
            console.print(
                f"[red]Error: No job directories (with config.json) found in: {job_path}[/red]"
            )
            raise SystemExit(1)
        console.print(f"Found [bold]{len(job_dirs)}[/bold] job(s) in {job_path}\n")

    for job_dir in job_dirs:
        if is_multi_job:
            # Skip jobs that already have summary.md unless --overwrite
            if (job_dir / "summary.md").exists() and not overwrite:
                console.print(
                    f"[dim]Skipping {job_dir.name} (summary.md already exists)[/dim]"
                )
                continue
            console.print(f"[bold blue]Summarizing {job_dir.name}...[/bold blue]")

        summarizer = Summarizer(
            job_dir,
            n_concurrent=n_concurrent,
            model=model,
            only_failed=not all_trials,
            overwrite=overwrite,
        )
        summary_path = summarizer.summarize()
        if summary_path:
            console.print(
                f"[green]✓ Summary complete![/green] View at: {summary_path}\n"
            )
        else:
            console.print(
                f"[yellow]⚠ No summary generated for {job_dir.name}[/yellow]\n"
            )


jobs_app.command()(start)
