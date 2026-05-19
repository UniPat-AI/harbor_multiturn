import hashlib
import json
import logging
import re
import shutil
import signal
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

import yaml
from dotenv import dotenv_values, load_dotenv
from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from harbor.agents.factory import AgentFactory
from harbor.cli.notifications import show_registry_hint_if_first_run
from harbor.cli.utils import load_mcp_servers, parse_env_vars, parse_kwargs, run_async
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import (
    DatasetConfig,
    JobConfig,
)
from harbor.models.job.lock import LOCK_FILENAME
from harbor.models.job.result import JobStats
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.multiround.resume_plan import (
    ResumeOutputMode,
    ResumeOutputPlan,
    plan_resume_output,
)

jobs_app = Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)
console = Console()
logger = logging.getLogger(__name__)

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


def _confirm_host_env_access(
    job,
    console: Console,
    *,
    explicit_env_file_keys: set[str] | None = None,
    skip_confirm: bool = False,
) -> None:
    import os
    import tomllib

    from pydantic import ValidationError

    from harbor.models.agent.name import AgentName
    from harbor.models.task.config import TaskConfig as TaskDefinitionConfig
    from harbor.models.task.paths import TaskPaths
    from harbor.utils.env import get_required_host_vars

    is_oracle = any(a.name == AgentName.ORACLE.value for a in job.config.agents)
    explicit_env_file_keys = explicit_env_file_keys or set()
    explicit_job_environment_keys = set(job.config.environment.env)
    explicit_job_verifier_keys = set(job.config.verifier.env)
    sections: dict[str, list[tuple[str, str | None]]] = {}

    for task_config in job._task_configs:
        try:
            local_path = task_config.get_local_path()
        except ValueError:
            continue
        if not local_path.exists():
            continue
        try:
            task_config_model = TaskDefinitionConfig.model_validate_toml(
                TaskPaths(local_path).config_path.read_text()
            )
        except (OSError, tomllib.TOMLDecodeError, ValidationError):
            continue

        env_sections = [
            ("environment", task_config_model.environment.env),
            ("verifier", task_config_model.verifier.env),
        ]
        if is_oracle:
            env_sections.append(("solution", task_config_model.solution.env))

        for section_name, env_dict in env_sections:
            filtered_env_dict = env_dict
            if section_name == "environment" and explicit_job_environment_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_job_environment_keys
                }
            elif section_name == "verifier" and explicit_job_verifier_keys:
                filtered_env_dict = {
                    key: value
                    for key, value in env_dict.items()
                    if key not in explicit_job_verifier_keys
                }

            required = [
                item
                for item in get_required_host_vars(filtered_env_dict)
                if item[0] not in explicit_env_file_keys
            ]
            if required:
                key = f"[{section_name}.env]"
                existing = sections.get(key, [])
                for item in required:
                    if item not in existing:
                        existing.append(item)
                if existing:
                    sections[key] = existing

    if not sections:
        return

    missing = []
    for section, vars_list in sections.items():
        for var_name, default in vars_list:
            if default is None and var_name not in os.environ:
                missing.append((section, var_name))

    if missing:
        table = Table(
            title="Missing Environment Variables",
            title_style="bold red",
            show_header=True,
            header_style="bold",
            padding=(0, 2),
            show_edge=False,
            show_lines=False,
        )
        table.add_column("Variable", style="cyan")
        table.add_column("Phase", style="dim")

        for section, var_name in missing:
            escaped = section.replace("[", "\\[")
            table.add_row(var_name, escaped)

        console.print()
        console.print(table)
        console.print(
            "\n[yellow]Export them in your shell or pass --env-file.[/yellow]"
        )
        raise SystemExit(1)

    if skip_confirm:
        return

    table = Table(
        title="Environment Variables",
        title_style="bold",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        show_edge=False,
        show_lines=False,
    )
    table.add_column("Variable", style="cyan")
    table.add_column("Phase", style="dim")

    for section, vars_list in sections.items():
        escaped = section.replace("[", "\\[")
        for var_name, default in vars_list:
            table.add_row(var_name, escaped)

    console.print()
    console.print(table)
    console.print()

    response = console.input(
        "Tasks in this run will load these from your environment. [yellow]Proceed? (Y/n):[/yellow] "
    )
    if response.strip().lower() in ("n", "no"):
        raise SystemExit(0)


def _format_duration(started_at: datetime | None, finished_at: datetime | None) -> str:
    if started_at is None or finished_at is None:
        return "unknown"

    total_seconds = max(0, int((finished_at - started_at).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_group_title(evals_key: str, job_result) -> str:
    parts = evals_key.split("__")
    if len(parts) == 3:
        agent_name, model_name, dataset_name = parts
    else:
        agent_name, dataset_name = parts
        model_name = None

    for trial_result in job_result.trial_results:
        trial_evals_key = JobStats.format_agent_evals_key(
            trial_result.agent_info.name,
            (
                trial_result.agent_info.model_info.name
                if trial_result.agent_info.model_info
                else None
            ),
            trial_result.source or "adhoc",
        )
        if trial_evals_key != evals_key:
            continue

        agent_name = trial_result.agent_info.name
        model_name = (
            trial_result.agent_info.model_info.name
            if trial_result.agent_info.model_info
            else model_name
        )
        dataset_name = trial_result.source or dataset_name
        break

    title_parts = [dataset_name, agent_name]
    if model_name:
        title_parts.append(model_name)
    return " • ".join(title_parts)


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
        console.print(f"[bold]{_format_group_title(evals_key, job_result)}[/bold]")

        summary_table = Table(show_header=True)
        summary_table.add_column("Trials", justify="right")
        summary_table.add_column("Exceptions", justify="right")

        summary_row = [str(dataset_stats.n_trials), str(dataset_stats.n_errors)]
        if dataset_stats.metrics:
            for i, metric in enumerate(dataset_stats.metrics):
                for key, value in metric.items():
                    metric_label = (
                        f"Metric {i + 1}: {key}"
                        if len(dataset_stats.metrics) > 1
                        else key
                    )
                    summary_table.add_column(metric_label.title(), justify="right")
                    if isinstance(value, float):
                        summary_row.append(f"{value:.3f}")
                    else:
                        summary_row.append(str(value))

        if dataset_stats.pass_at_k:
            for k, value in sorted(dataset_stats.pass_at_k.items()):
                summary_table.add_column(f"Pass@{k}", justify="right")
                summary_row.append(f"{value:.3f}")

        summary_table.add_row(*summary_row)
        console.print(summary_table)

        if dataset_stats.reward_stats:
            reward_table = Table(show_header=True)
            reward_table.add_column("Reward")
            reward_table.add_column("Count", justify="right")
            round_keys = sorted(
                [
                    reward_key
                    for reward_key in dataset_stats.reward_stats
                    if re.fullmatch(r"round_\d+", reward_key)
                ],
                key=_round_key_order,
            )
            has_multiround_rewards = len(round_keys) > 0

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
                label = (
                    "final_reward"
                    if reward_key == "reward" and has_multiround_rewards
                    else reward_key
                )

                if has_multiround_rewards:
                    total_count = sum(
                        len(trial_names) for trial_names in reward_values.values()
                    )
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
                        reward_table.add_row(
                            f"{label} mean",
                            f"{mean_reward:.3f} (n={total_count})",
                        )

                        is_binary_reward = all(
                            value in (0.0, 1.0) for value, _ in numeric_entries
                        )
                        if is_binary_reward:
                            success_count = sum(
                                count
                                for value, count in numeric_entries
                                if value == 1.0
                            )
                            success_rate = success_count / total_count
                            reward_table.add_row(
                                f"{label} success",
                                f"{success_count}/{total_count} ({success_rate:.1%})",
                            )
                        continue

                    bins = ", ".join(
                        f"{reward_value}:{len(trial_names)}"
                        for reward_value, trial_names in sorted(
                            reward_values.items(),
                            key=lambda item: _reward_value_sort_key(item[0]),
                        )
                    )
                    reward_table.add_row(f"{label} bins", bins if bins else "-")
                    continue

                for reward_value, trial_names in reward_values.items():
                    count = len(trial_names)
                    reward_table.add_row(f"{label} = {reward_value}", str(count))
            console.print()
            console.print(reward_table)

        if dataset_stats.exception_stats:
            exception_table = Table(show_header=True)
            exception_table.add_column("Exception")
            exception_table.add_column("Count", justify="right")
            for (
                exception_type,
                trial_names,
            ) in dataset_stats.exception_stats.items():
                count = len(trial_names)
                exception_table.add_row(exception_type, str(count))
            console.print()
            console.print(exception_table)

        console.print()


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
    """Clean an existing trial directory before writing an in-place resume."""
    for name in ("result.json", "config.json", "trial.log", "exception.txt"):
        path = trial_dir / name
        if path.exists():
            path.unlink()

    verifier_dir = trial_dir / "verifier"
    if verifier_dir.exists():
        for item in verifier_dir.iterdir():
            if item.name.startswith("round_"):
                parts = item.name.split("_", 2)
                if len(parts) >= 2:
                    try:
                        if int(parts[1]) >= from_round:
                            item.unlink()
                    except ValueError:
                        pass

        for name in (
            "multiround_results.json",
            "reward.txt",
            "reward.json",
            "test-stdout.txt",
            "test-stderr.txt",
            "test-exit-code.txt",
            "test_output.txt",
            "verification-error.txt",
        ):
            (verifier_dir / name).unlink(missing_ok=True)

    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        for item in agent_dir.iterdir():
            name = item.name
            if name.startswith("round-") and "-command-" in name:
                try:
                    round_num = int(name.split("-")[1])
                    if round_num >= from_round:
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
                except (ValueError, IndexError):
                    pass
                continue

            if name.startswith("claude-code-round-"):
                try:
                    round_num = int(name.replace("claude-code-round-", "").split(".")[0])
                    if round_num >= from_round:
                        item.unlink()
                except (ValueError, IndexError):
                    pass
                continue

            if name.startswith("round_") and (
                name.endswith("_oracle.txt") or name.endswith("_exit-code.txt")
            ):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    try:
                        if int(parts[1]) >= from_round:
                            item.unlink(missing_ok=True)
                    except ValueError:
                        pass
                continue

            if name in ("session_snapshots", "runtime_snapshots") and item.is_dir():
                for entry in item.iterdir():
                    if not entry.is_dir() or not entry.name.startswith("round_"):
                        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
                        continue
                    try:
                        round_num = int(entry.name.split("_", 1)[1])
                    except ValueError:
                        shutil.rmtree(entry, ignore_errors=True)
                        continue
                    if round_num >= from_round:
                        shutil.rmtree(entry, ignore_errors=True)
                continue

            if name in ("install.sh", "setup"):
                continue
            if name == "claude-code.txt" and from_round > 1:
                continue

            shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(
                missing_ok=True
            )

    state_dir = trial_dir / "state"
    if state_dir.exists():
        preserved_rounds: list[int] = []

        for name in ("snapshot.json", "snapshot-image.tar"):
            path = state_dir / name
            if path.exists() or path.is_symlink():
                path.unlink()

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
    config_path = resume_trial_dir / "config.json"
    if not config_path.exists():
        raise ValueError(
            "--resume-trial requires an existing config.json in the source trial: "
            f"{resume_trial_dir}"
        )

    try:
        return TrialConfig.model_validate_json(config_path.read_text())
    except Exception as exc:
        raise ValueError(
            f"Failed to read source trial config from {config_path}: {exc}"
        ) from exc


def _load_resume_source_task_checksum(resume_trial_dir: Path) -> str | None:
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

    return relative_path.name in _VOLATILE_TASK_FILE_NAMES


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


def _iter_resume_lineage_trial_dirs(resume_trial_dir: Path):
    seen: set[Path] = set()
    current = resume_trial_dir.expanduser()

    while True:
        try:
            resolved = current.resolve()
        except OSError:
            resolved = current.absolute()
        if resolved in seen:
            return

        seen.add(resolved)
        yield current

        config_path = current / "config.json"
        if not config_path.exists():
            return
        try:
            payload = json.loads(config_path.read_text())
        except Exception:
            return

        verifier = payload.get("verifier")
        if not isinstance(verifier, dict):
            return
        parent = verifier.get("multiround_resume_source")
        if not isinstance(parent, str) or not parent.strip():
            return
        current = Path(parent).expanduser()


def _resolve_resume_snapshot_metadata_path(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    completed_round = start_round - 1

    for candidate_dir in _iter_resume_lineage_trial_dirs(resume_trial_dir):
        trial_paths = TrialPaths(candidate_dir)

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
            continue

        latest_snapshot_path = trial_paths.state_snapshot_path
        if latest_snapshot_path.exists():
            return latest_snapshot_path
    return None


def _resolve_resume_claude_sessions_dir(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    completed_round = start_round - 1
    if completed_round < 1:
        return None

    for candidate_dir in _iter_resume_lineage_trial_dirs(resume_trial_dir):
        trial_paths = TrialPaths(candidate_dir)
        round_sessions_dir = trial_paths.agent_round_sessions_dir(completed_round)
        if round_sessions_dir.exists():
            return round_sessions_dir

        latest_sessions_dir = trial_paths.agent_sessions_dir
        if not latest_sessions_dir.exists():
            continue

        latest_round = _resolve_resume_source_latest_round(candidate_dir)
        if latest_round == completed_round:
            return latest_sessions_dir
    return None


def _resolve_resume_terminus_runtime_state_path(
    resume_trial_dir: Path, *, start_round: int
) -> Path | None:
    completed_round = start_round - 1
    if completed_round < 1:
        return None

    for candidate_dir in _iter_resume_lineage_trial_dirs(resume_trial_dir):
        trial_paths = TrialPaths(candidate_dir)
        round_state_path = trial_paths.terminus_round_runtime_state_path(
            completed_round
        )
        if round_state_path.exists():
            return round_state_path
    return None


def _validate_resume_task_identity(
    target_task: Task,
    target_task_config: TaskConfig,
    resume_trial_dir: Path,
) -> None:
    source_trial_config = _load_resume_source_trial_config(resume_trial_dir)

    source_task_path = source_trial_config.task.path
    if source_task_path is not None and source_task_path.exists():
        source_checksum = _compute_task_definition_checksum(source_task_path)
        target_checksum = _compute_task_definition_checksum(target_task.task_dir)
        if source_checksum != target_checksum:
            raise ValueError(
                "--resume-trial source task does not match the current task. "
                "Resume requires the same task definition."
            )
        return

    source_checksum = _load_resume_source_task_checksum(resume_trial_dir)
    if source_checksum is None and source_trial_config.task.path is not None:
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
    except Exception as exc:
        raise ValueError(
            "--resume-trial preflight failed: could not read resume snapshot metadata "
            f"from {snapshot_meta_path}: {exc}"
        ) from exc

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
    target_agent_name = AgentFactory.get_agent_class_from_config(
        target_agent_config
    ).name()

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


def _print_multiround_resume_dry_run_report(
    *,
    task: Task,
    resolved_resume: Path,
    resume_round: int,
    resume_plan: ResumeOutputPlan,
    preflight_policy: str,
) -> None:
    planned_resume_source = (
        resume_plan.backup_dir("<timestamp>")
        if resume_plan.uses_backup
        else resume_plan.resume_source_dir
    )

    rows = [
        ("task", task.name),
        ("resume_round", str(resume_round)),
        ("resolved_resume_trial", str(resolved_resume)),
        ("output_mode", resume_plan.output_mode.value),
        ("jobs_dir", str(resume_plan.jobs_dir)),
        ("job_name", resume_plan.job_name or "<auto>"),
        ("preflight_policy", preflight_policy),
        ("would_mutate_source_trial", str(resume_plan.mutates_source_trial)),
        ("would_create_backup", str(resume_plan.uses_backup)),
        ("planned_resume_source", str(planned_resume_source)),
        (
            "config_resume_trial_name",
            resume_plan.resume_trial_name_for_config or "<none>",
        ),
    ]
    console.print("Multi-round resume dry run")
    for field, value in rows:
        console.print(f"{field}: {value}")


def _would_enable_roundwise_multiround_attempt_selection(config: JobConfig) -> bool:
    if (config.jobs_dir / config.job_name / "result.json").exists():
        return False
    if config.verifier.multiround_resume_trial_name is not None:
        return False

    task_configs = list(config.tasks)
    if not task_configs or config.datasets:
        return False

    for task_config in task_configs:
        if task_config.is_git_task() or task_config.path is None:
            return False
        if not Task(task_config.path).is_multiround:
            return False

    return True


def _harbor_hub_visibility(public: bool | None):
    """Translate the tri-state CLI flag into the Uploader's visibility arg.

    ``--public`` → ``"public"``; ``--private`` → ``"private"``; no flag →
    ``None`` (defer to Uploader's "default private for new, unchanged for
    existing" logic).
    """
    if public is None:
        return None
    return "public" if public else "private"


async def _setup_harbor_hub_streaming(
    job,
    *,
    public: bool | None,
    share_orgs: list[str] | None = None,
    share_users: list[str] | None = None,
    confirm_non_member_orgs: bool = False,
):
    """Insert the job row at run start and register the per-trial streaming
    hook on ``job``. Returns the Uploader (so the post-run finalize can
    reuse the same instance) and the JobStartResult (so the hook closure
    has the caches it needs).

    Failure handling distinguishes two cases:
      * **Auth failure** (the user isn't logged in or the token expired) →
        hard exit, since `--upload` clearly says they want to upload and
        running the entire job only to fail at end is the wrong UX.
      * **Anything else** (transient network error, RPC blip, etc.) → log a
        warning and return ``(None, None)``. The run still proceeds; the
        post-run :func:`_run_harbor_hub_upload` will do the full batch
        upload at end, which is equivalent (just slower).
    """
    from harbor.trial.hooks import TrialHookEvent
    from harbor.upload.uploader import Uploader

    visibility = _harbor_hub_visibility(public)
    uploader = Uploader()
    try:
        job_start = await uploader.start_job(
            job_id=job.id,
            job_name=job.config.job_name,
            started_at=datetime.now(),
            config=job.config.model_dump(mode="json"),
            visibility=visibility,
            share_orgs=share_orgs,
            share_users=share_users,
            confirm_non_member_orgs=confirm_non_member_orgs,
            # `len(job)` returns the configured trial count — what the
            # orchestrator was asked to run. The viewer pairs this with
            # `n_completed` from the trial table to show progress while
            # the run is still in flight.
            n_planned_trials=len(job),
        )
    except RuntimeError as exc:
        # `UploadDB.get_user_id` raises RuntimeError("Not authenticated...")
        # on missing/invalid auth. That's a hard fail-fast for --upload.
        if "Not authenticated" in str(exc):
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1) from None
        console.print(
            f"[yellow]Warning:[/yellow] Could not register job with Harbor Hub "
            f"at start: {type(exc).__name__}: {exc}. Will batch-upload at end.",
            soft_wrap=True,
        )
        return None, None
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Could not register job with Harbor Hub "
            f"at start: {type(exc).__name__}: {exc}. Will batch-upload at end.",
            soft_wrap=True,
        )
        return None, None

    async def _streaming_upload_cb(event: TrialHookEvent) -> None:
        # END events carry the result; all other lifecycle events are noise
        # for our purposes (we only care once result.json is on disk).
        if event.result is None:
            return
        # `trial_dir` matches what `Trial.trial_dir` computes — the same
        # directory the orchestrator wrote `result.json` + `agent/` etc.
        # to, which `Uploader._upload_single_trial` knows how to tar.
        trial_dir = event.config.trials_dir / event.config.trial_name
        try:
            await uploader.upload_single_trial(
                trial_result=event.result,
                trial_dir=trial_dir,
                job_id=job_start.job_id,
                agent_cache=job_start.agent_cache,
                model_cache=job_start.model_cache,
            )
        except Exception as exc:
            # Streaming failures are non-fatal: the trial is on disk, and
            # the post-run sweep (`upload_job`) is idempotent so it will
            # pick up any trial that didn't make it.
            logger.debug(
                "Trial %s failed to upload during run: %s. "
                "Will retry at end-of-run finalize.",
                event.result.trial_name,
                exc,
            )

    job.on_trial_ended(_streaming_upload_cb)
    return uploader, job_start


async def _run_harbor_hub_upload(
    job_dir: Path,
    *,
    public: bool | None,
    share_orgs: list[str] | None = None,
    share_users: list[str] | None = None,
    confirm_non_member_orgs: bool = False,
    yes: bool = False,
) -> None:
    """Post-run finalize for `harbor run --upload`.

    Calls the idempotent ``Uploader.upload_job`` sweep — for a streamed run
    this is just "find no missing trials, build job archive, finalize." For
    a run where streaming setup failed (or wasn't enabled), this is the
    full batch upload. Either way, the end state is the same.

    Async-only because it must share the event loop with the run itself —
    ``run_async`` (asyncio.run) creates and tears down a loop per call, and
    the Supabase auth client caches a singleton tied to whichever loop
    created it. Cross-loop reuse raises ``Event loop is closed``. Calling
    this from inside the same `_run_job` coroutine that did the streaming
    setup keeps everything on one loop.

    Isolates the upload from the run — a failed upload doesn't mask the
    fact that the job itself ran fine, and the user gets a retry command.
    """
    from harbor.cli.job_sharing import format_share_summary, retry_share_flags
    from harbor.constants import HARBOR_VIEWER_JOBS_URL
    from harbor.upload.uploader import Uploader

    visibility = _harbor_hub_visibility(public)

    try:
        uploader = Uploader()
        result = await uploader.upload_job(
            job_dir,
            visibility=visibility,
            share_orgs=share_orgs,
            share_users=share_users,
            confirm_non_member_orgs=confirm_non_member_orgs,
        )
        console.print(
            f"Uploaded to Harbor Hub: "
            f"{HARBOR_VIEWER_JOBS_URL}/{result.job_id} "
            f"(visibility: {result.visibility})"
        )
        share_summary = format_share_summary(
            share_orgs=result.shared_orgs
            if isinstance(result.shared_orgs, list)
            else [],
            share_users=result.shared_users
            if isinstance(result.shared_users, list)
            else [],
        )
        if share_summary:
            console.print(f"Shared with {share_summary}")
    except Exception as exc:
        # Don't mask the fact that the run itself succeeded. Point to the
        # retry command and include the specific visibility flag the caller
        # asked for (if any) so they can copy-paste.
        retry_flag = (
            " --public" if public is True else " --private" if public is False else ""
        )
        retry_flag += retry_share_flags(
            share_orgs=share_orgs,
            share_users=share_users,
            yes=yes,
        )
        console.print(
            f"[yellow]Warning:[/yellow] Job completed but upload failed: "
            f"{type(exc).__name__}: {exc}"
        )
        # soft_wrap=True keeps the retry command on a single line so users
        # can copy-paste it without having to stitch together a wrapped path.
        console.print(
            f"Retry with `harbor upload {job_dir}{retry_flag}`", soft_wrap=True
        )


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
            help=f"Directory to store job results (default: {
                JobConfig.model_fields['jobs_dir'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    n_attempts: Annotated[
        int | None,
        Option(
            "-k",
            "--n-attempts",
            help=f"Number of attempts per trial (default: {
                JobConfig.model_fields['n_attempts'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    timeout_multiplier: Annotated[
        float | None,
        Option(
            "--timeout-multiplier",
            help=f"Multiplier for task timeouts (default: {
                JobConfig.model_fields['timeout_multiplier'].default
            })",
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
    n_concurrent_trials: Annotated[
        int | None,
        Option(
            "-n",
            "--n-concurrent",
            help=f"Number of concurrent trials to run (default: {
                JobConfig.model_fields['n_concurrent_trials'].default
            })",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    max_retries: Annotated[
        int | None,
        Option(
            "-r",
            "--max-retries",
            help="Maximum number of retry attempts (default: 0)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_include_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-include",
            help="Exception types to retry on. If not specified, all exceptions except "
            "those in --retry-exclude are retried (can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
    retry_exclude_exceptions: Annotated[
        list[str] | None,
        Option(
            "--retry-exclude",
            help="Exception types to NOT retry on (can be used multiple times)",
            rich_help_panel="Job Settings",
            show_default=False,
        ),
    ] = None,
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
    mcp_config: Annotated[
        list[Path] | None,
        Option(
            "--mcp-config",
            help="Path to a Claude-style .mcp.json or Harbor MCP config file. Can be used multiple times.",
            rich_help_panel="Agent",
            show_default=False,
        ),
    ] = None,
    skills: Annotated[
        list[Path] | None,
        Option(
            "--skill",
            "--skills",
            help="Path to a skill directory, or a root containing skill directories. "
            "Can be used multiple times.",
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
            help=f"Whether to force rebuild the environment (default: {
                '--force-build'
                if EnvironmentConfig.model_fields['force_build'].default
                else '--no-force-build'
            })",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    environment_delete: Annotated[
        bool | None,
        Option(
            "--delete/--no-delete",
            help=f"Whether to delete the environment after completion (default: {
                '--delete'
                if EnvironmentConfig.model_fields['delete'].default
                else '--no-delete'
            })",
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
    mounts: Annotated[
        str | None,
        Option(
            "--mounts",
            "--mounts-json",
            help="JSON array of volume mounts for the environment container "
            "(Docker Compose service volume format). --mounts-json is a "
            "deprecated alias.",
            rich_help_panel="Environment",
            show_default=False,
        ),
    ] = None,
    extra_docker_compose: Annotated[
        list[Path] | None,
        Option(
            "--extra-docker-compose",
            help="Additional Docker Compose overlay file. Can be used multiple times.",
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
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help=(
                "Auto-confirm prompts, including host environment access and "
                "sharing with organizations you are not a member of."
            ),
            rich_help_panel="Job Settings",
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        Option(
            "--env-file",
            help="Path to a .env file to load into environment.",
            rich_help_panel="Job Settings",
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
    extra_instruction_paths: Annotated[
        list[Path] | None,
        Option(
            "--extra-instruction-path",
            help="Path to an extra instruction file to append to the task "
            "instruction. Can be used multiple times.",
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
    task_ref: Annotated[
        str | None,
        Option(
            "-t",
            "--task",
            help="Run a single task from the registry (org/name[@ref])",
            rich_help_panel="Dataset",
            show_default=False,
        ),
    ] = None,
    dataset_task_names: Annotated[
        list[str] | None,
        Option(
            "-i",
            "--include-task-name",
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
            hidden=True,
        ),
    ] = False,
    export_sharegpt: Annotated[
        bool,
        Option(
            "--export-sharegpt/--no-export-sharegpt",
            help="Also emit ShareGPT column when exporting traces",
            hidden=True,
        ),
    ] = False,
    export_episodes: Annotated[
        str,
        Option(
            "--export-episodes",
            help="Which episodes to export per trial (all|last)",
            hidden=True,
        ),
    ] = "all",
    export_push: Annotated[
        bool,
        Option(
            "--export-push/--no-export-push",
            help="Push exported dataset to HF Hub",
            hidden=True,
        ),
    ] = False,
    export_repo: Annotated[
        str | None,
        Option(
            "--export-repo",
            help="Target HF repo id (org/name) when pushing traces",
            hidden=True,
        ),
    ] = None,
    export_instruction_metadata: Annotated[
        bool,
        Option(
            "--export-instruction-metadata/--no-export-instruction-metadata",
            help="Include instruction text column when exporting traces",
            hidden=True,
        ),
    ] = False,
    export_verifier_metadata: Annotated[
        bool,
        Option(
            "--export-verifier-metadata/--no-export-verifier-metadata",
            help="Include verifier stdout/stderr column when exporting traces",
            hidden=True,
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
    verifier_env: Annotated[
        list[str] | None,
        Option(
            "--ve",
            "--verifier-env",
            help="Environment variable to pass to the verifier in KEY=VALUE format. "
            "Can be used multiple times. Example: --ve OPENAI_BASE_URL=http://localhost:8000/v1",
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
            help="In multi-round mode, number of successful parent trajectories "
            "selected per round to continue to the next round.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_state_cache_policy: Annotated[
        str | None,
        Option(
            "--multiround-state-cache-policy",
            help="Container state snapshot caching policy: off, success, or all.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_resume_preflight_policy: Annotated[
        str | None,
        Option(
            "--multiround-resume-preflight-policy",
            help="Resume preflight policy for multi-round resume: off, snapshot, or strict.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    max_round: Annotated[
        int | None,
        Option(
            "--max-round",
            help="Maximum round number to execute.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_aggregate_start_round: Annotated[
        int | None,
        Option(
            "--multiround-aggregate-start-round",
            help="Override the first round included in final multi-round reward aggregation.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    multiround_aggregate_end_round: Annotated[
        int | None,
        Option(
            "--multiround-aggregate-end-round",
            help="Override the last round included in final multi-round reward aggregation.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    start_round: Annotated[
        int | None,
        Option(
            "--start-round",
            help="Start from round N after fast-forwarding earlier rounds with oracle solutions.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    resume_trial: Annotated[
        Path | None,
        Option(
            "--resume-trial",
            help="Existing trial or job directory to resume from.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    resume_round: Annotated[
        int | None,
        Option(
            "--resume-round",
            help="Round number to resume from; requires --resume-trial.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    no_resume_backup: Annotated[
        bool,
        Option(
            "--no-resume-backup",
            help="Skip creating __resumed_<timestamp> backup dirs for in-place resume.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = False,
    output_jobs_dir: Annotated[
        Path | None,
        Option(
            "--output-jobs-dir",
            help="Write resume output to this jobs directory instead of modifying the source trial.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = None,
    resume_dry_run: Annotated[
        bool,
        Option(
            "--resume-dry-run",
            help="Validate and print the multi-round resume plan without mutating trials or starting a job.",
            rich_help_panel="Multi-round",
            show_default=False,
        ),
    ] = False,
    upload: Annotated[
        bool,
        Option(
            "--upload",
            help="After the job finishes, upload it to Harbor Hub so you can "
            "share the run via a link.",
            rich_help_panel="Harbor Hub",
        ),
    ] = False,
    public: Annotated[
        bool | None,
        Option(
            "--public/--private",
            help="Visibility for the uploaded job. Requires --upload. "
            "No flag = private (default).",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
    share_org: Annotated[
        list[str] | None,
        Option(
            "--share-org",
            help="Share the uploaded job with an organization. Requires --upload. Repeatable.",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
    share_user: Annotated[
        list[str] | None,
        Option(
            "--share-user",
            help="Share the uploaded job with a GitHub username. Requires --upload. Repeatable.",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
):
    from harbor.job import Job

    # Harbor Hub flag validation: --public/--private requires --upload so the
    # semantics stay explicit (no hidden "oh, you wanted to upload too").
    if public is not None and not upload:
        console.print("[red]Error:[/red] --public / --private requires --upload.")
        raise SystemExit(1)
    if (share_org or share_user) and not upload:
        console.print("[red]Error:[/red] --share-org / --share-user requires --upload.")
        raise SystemExit(1)
    if resume_dry_run and upload:
        raise ValueError("--resume-dry-run does not support --upload")

    if env_file is not None:
        if not env_file.exists():
            console.print(f"[red]❌ Env file not found: {env_file}[/red]")
            raise SystemExit(1)
        load_dotenv(env_file, override=True)

    # Auth check for --upload happens inside `_setup_harbor_hub_streaming`,
    # which is called at the start of `_run_job` (before any orchestrator
    # work). We deliberately don't pre-check here in a separate `run_async`
    # call — `run_async` calls `asyncio.run`, which creates and tears down
    # an event loop per invocation. The Supabase auth client caches a
    # singleton tied to the loop that created it, so a cross-loop reuse
    # raises ``Event loop is closed``. Doing the check inside the same
    # `run_async` as the run itself avoids that.

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

    if n_concurrent_trials is not None:
        config.n_concurrent_trials = n_concurrent_trials
    if quiet:
        config.quiet = quiet
    if max_retries is not None:
        config.retry.max_retries = max_retries
    if retry_include_exceptions is not None:
        config.retry.include_exceptions = set(retry_include_exceptions)
    if retry_exclude_exceptions is not None:
        config.retry.exclude_exceptions = set(retry_exclude_exceptions)

    if agent_name is not None or agent_import_path is not None:
        config.agents = []
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        parsed_mcp_servers = [
            server
            for mcp_config_path in mcp_config or []
            for server in load_mcp_servers(mcp_config_path)
        ]

        if model_names is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
                for model_name in model_names
            ]
        else:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    skills=skills or [],
                    kwargs=parsed_kwargs,
                    env=parsed_env,
                    mcp_servers=parsed_mcp_servers,
                )
            ]
    else:
        parsed_kwargs = parse_kwargs(agent_kwargs)
        parsed_env = parse_env_vars(agent_env)
        parsed_mcp_servers = [
            server
            for mcp_config_path in mcp_config or []
            for server in load_mcp_servers(mcp_config_path)
        ]
        if parsed_kwargs or parsed_env or parsed_mcp_servers or skills:
            for agent in config.agents:
                if parsed_kwargs:
                    agent.kwargs.update(parsed_kwargs)
                if parsed_env:
                    agent.env.update(parsed_env)
                if parsed_mcp_servers:
                    agent.mcp_servers.extend(parsed_mcp_servers)
                if skills:
                    agent.skills.extend(skills)

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
    if mounts is not None:
        config.environment.mounts = json.loads(mounts)
    if extra_docker_compose is not None:
        config.environment.extra_docker_compose.extend(extra_docker_compose)
    if environment_kwargs is not None:
        config.environment.kwargs.update(parse_kwargs(environment_kwargs))

    if verifier_env is not None:
        config.verifier.env.update(parse_env_vars(verifier_env))
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
        pending_resume = (resume_trial, resume_round)

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
    if (
        config.verifier.multiround_max_round is not None
        and effective_start > config.verifier.multiround_max_round
    ):
        raise ValueError(
            f"--start-round ({effective_start}) must be <= "
            f"--max-round ({config.verifier.multiround_max_round})"
        )

    if artifact_paths is not None:
        config.artifacts = list(artifact_paths)
    if extra_instruction_paths is not None:
        config.extra_instruction_paths = list(extra_instruction_paths)

    task_specified = task_git_url is not None or task_git_commit_id is not None

    dataset_specified = (
        dataset_name_version is not None
        or registry_url is not None
        or registry_path is not None
        or dataset_task_names is not None
        or dataset_exclude_task_names is not None
        or n_tasks is not None
        or task_ref is not None
    )

    if task_specified and dataset_specified:
        raise ValueError("Cannot specify both task and dataset parameters")

    if path is not None:
        is_task = Task.is_valid_dir(
            path,
            disable_verification=disable_verification,
        )

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
                DatasetConfig(
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

        if task_ref is not None:
            from harbor.models.package.reference import PackageReference

            ref = PackageReference.parse(task_ref)
            config.datasets = []
            config.tasks = [
                TaskConfig(
                    name=ref.name,
                    ref=ref.ref,
                )
            ]

        elif dataset_name_version is not None:
            if "@" in dataset_name_version:
                name, version = dataset_name_version.split("@", 1)
            else:
                name, version = dataset_name_version, None

            # Detect package reference (org/name format)
            if "/" in name:
                config.datasets = [
                    DatasetConfig(
                        name=name,
                        ref=version or "latest",
                        task_names=dataset_task_names,
                        exclude_task_names=dataset_exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]
            else:
                config.datasets = [
                    DatasetConfig(
                        registry_url=registry_url,
                        registry_path=registry_path,
                        name=name,
                        version=version,
                        task_names=dataset_task_names,
                        exclude_task_names=dataset_exclude_task_names,
                        n_tasks=n_tasks,
                    )
                ]

        else:
            raise ValueError(
                "Cannot specify --registry-url, --registry-path, --include-task-name, or "
                "--exclude-task-name without also specifying --dataset, --task, or --path."
            )

    task_configs = list(config.tasks)
    if config.verifier.multiround_start_round is not None:
        start_round_value = config.verifier.multiround_start_round
        for task_config in task_configs:
            if task_config.is_git_task() or task_config.path is None:
                continue
            task = Task(task_config.path)
            if start_round_value > task.num_rounds:
                raise ValueError(
                    f"--start-round ({start_round_value}) must be <= "
                    f"the task's num_rounds ({task.num_rounds}) for task '{task.name}'"
                )

    for task_config in task_configs:
        if task_config.is_git_task() or task_config.path is None:
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
        raise ValueError("--output-jobs-dir requires --resume-trial")
    if resume_dry_run and pending_resume is None:
        raise ValueError("--resume-dry-run requires --resume-trial")

    if pending_resume is not None:
        resume_trial_path, resolved_resume_round = pending_resume

        _validate_resume_shape(config)

        if config.datasets:
            raise ValueError(
                "--resume-trial is not supported when running a dataset or multiple "
                "tasks; resume requires a single task job"
            )

        if len(config.tasks) != 1 or config.tasks[0].path is None:
            raise ValueError(
                "--resume-trial requires exactly one local task in the job configuration"
            )

        resolved_resume = _resolve_resume_trial_dir(
            resume_trial_path,
            start_round=resolved_resume_round,
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

        resume_plan = plan_resume_output(
            resolved_resume=resolved_resume,
            jobs_dir=jobs_dir,
            output_jobs_dir=output_jobs_dir,
            no_resume_backup=no_resume_backup,
        )

        config.jobs_dir = resume_plan.jobs_dir
        if resume_plan.job_name is not None:
            config.job_name = resume_plan.job_name
        resume_source_dir = resume_plan.resume_source_dir
        resume_trial_name_for_config = resume_plan.resume_trial_name_for_config

        if resume_dry_run:
            _print_multiround_resume_dry_run_report(
                task=task,
                resolved_resume=resolved_resume,
                resume_round=resolved_resume_round,
                resume_plan=resume_plan,
                preflight_policy=config.verifier.multiround_resume_preflight_policy,
            )
            return

        if resume_plan.output_mode != ResumeOutputMode.COPY_TO_OUTPUT_JOBS_DIR:
            if resume_plan.output_mode == ResumeOutputMode.IN_PLACE_WITHOUT_BACKUP:
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
                backup_dir = resume_plan.backup_dir(ts)
                shutil.copytree(resolved_resume, backup_dir, symlinks=True)
                _cleanup_trial_for_resume(resolved_resume, resolved_resume_round)
                resume_source_dir = backup_dir

            for name in ("result.json", "config.json", LOCK_FILENAME):
                path_to_remove = resume_plan.trial_parent / name
                if path_to_remove.exists():
                    path_to_remove.unlink()

        config.verifier.multiround_start_round = resolved_resume_round
        config.verifier.multiround_resume_source = str(resume_source_dir)
        config.verifier.multiround_resume_trial_name = resume_trial_name_for_config

    if (
        config.verifier.multiround_state_cache_policy == "off"
        and _would_enable_roundwise_multiround_attempt_selection(config)
    ):
        raise ValueError(
            "Multi-round round-wise execution with --multiround-state-cache-policy off "
            "is unsupported: child trials require parent snapshots which are never "
            "saved under policy 'off'. "
            "Use 'success' (default) or 'all' instead."
        )

    async def _run_job():
        from harbor.cli.job_sharing import (
            confirm_non_member_org_shares,
            normalize_share_values,
        )

        requested_share_orgs = normalize_share_values(share_org)
        requested_share_users = normalize_share_values(share_user)
        confirm_non_member_orgs = (
            await confirm_non_member_org_shares(requested_share_orgs, yes=yes)
            if upload
            else False
        )
        job = await Job.create(config)
        _confirm_host_env_access(
            job,
            console,
            explicit_env_file_keys=explicit_env_file_keys,
            skip_confirm=yes,
        )
        # Streaming Harbor Hub upload: insert the job row + register the
        # per-trial END hook before the orchestrator starts. Trials upload
        # as they finish; the post-run finalize call below sweeps anything
        # the hook missed and writes the job-level archive.
        if upload:
            await _setup_harbor_hub_streaming(
                job,
                public=public,
                share_orgs=requested_share_orgs,
                share_users=requested_share_users,
                confirm_non_member_orgs=confirm_non_member_orgs,
            )
        job_result = await job.run()

        # Print the run summary BEFORE finalize so users see results even
        # if the upload retries / takes a moment.
        console.print()
        print_job_results_tables(job_result)
        console.print("[bold]Job Info[/bold]")
        console.print(
            "Total runtime: "
            f"{_format_duration(getattr(job_result, 'started_at', None), getattr(job_result, 'finished_at', None))}"
        )
        console.print(
            f"Results written to {getattr(job, '_job_result_path', job.job_dir / 'result.json')}"
        )
        console.print(f"Inspect results by running `harbor view {job.job_dir.parent}`")

        # Finalize must run on the same event loop as `_setup_harbor_hub_streaming`
        # — the Supabase auth client is a module-level singleton tied to
        # whichever loop first created it. A separate `run_async` call would
        # create a new loop and reuse the dead-loop-bound client → boom.
        if upload:
            await _run_harbor_hub_upload(
                job.job_dir,
                public=public,
                share_orgs=requested_share_orgs,
                share_users=requested_share_users,
                confirm_non_member_orgs=confirm_non_member_orgs,
                yes=yes,
            )
        else:
            console.print(f"Share results by running `harbor upload {job.job_dir}`")
        console.print()

        return job, job_result

    from harbor.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    explicit_env_file_keys: set[str] = set()
    if env_file is not None:
        explicit_env_file_keys = {
            key for key in dotenv_values(env_file).keys() if key is not None
        }

    signal.signal(signal.SIGTERM, _handle_sigterm)

    show_registry_hint_if_first_run(console)

    # `_run_job` itself prints the summary + invokes the upload finalize
    # (when --upload is set) so everything stays on one event loop. See
    # the long comment in `_run_harbor_hub_upload` for why this matters.
    job, job_result = run_async(_run_job())

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
    upload: Annotated[
        bool,
        Option(
            "--upload",
            help="After the resume finishes, upload the job to Harbor Hub. "
            "If the job was already partially uploaded (e.g. from a previous "
            "`harbor run --upload` that crashed), this fills in the missing "
            "trials and finalizes — it's an idempotent sweep.",
            rich_help_panel="Harbor Hub",
        ),
    ] = False,
    public: Annotated[
        bool | None,
        Option(
            "--public/--private",
            help="Visibility for the uploaded job. Requires --upload. "
            "No flag = private for new jobs, unchanged for existing ones.",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
    share_org: Annotated[
        list[str] | None,
        Option(
            "--share-org",
            help="Share the uploaded job with an organization. Requires --upload. Repeatable.",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
    share_user: Annotated[
        list[str] | None,
        Option(
            "--share-user",
            help="Share the uploaded job with a GitHub username. Requires --upload. Repeatable.",
            rich_help_panel="Harbor Hub",
            show_default=False,
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Confirm sharing with organizations you are not a member of.",
            rich_help_panel="Harbor Hub",
        ),
    ] = False,
):
    """Resume an existing job from its job directory."""
    from harbor.job import Job

    # Mirror the validation in `start()` — setting visibility without
    # --upload is almost certainly a mistake, so fail fast rather than
    # silently ignoring the flag.
    if public is not None and not upload:
        console.print("[red]Error:[/red] --public / --private requires --upload.")
        raise SystemExit(1)
    if (share_org or share_user) and not upload:
        console.print("[red]Error:[/red] --share-org / --share-user requires --upload.")
        raise SystemExit(1)

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
                    f"Removing trial directory with {
                        trial_result.exception_info.exception_type
                    }: {trial_dir.name}"
                )
                shutil.rmtree(trial_dir)

    config = JobConfig.model_validate_json(config_path.read_text())

    from harbor.environments.factory import EnvironmentFactory

    EnvironmentFactory.run_preflight(
        type=config.environment.type,
        import_path=config.environment.import_path,
    )

    async def _run_job():
        from harbor.cli.job_sharing import (
            confirm_non_member_org_shares,
            normalize_share_values,
        )

        requested_share_orgs = normalize_share_values(share_org)
        requested_share_users = normalize_share_values(share_user)
        confirm_non_member_orgs = (
            await confirm_non_member_org_shares(requested_share_orgs, yes=yes)
            if upload
            else False
        )
        job = await Job.create(config)
        # Same streaming setup as `harbor run --upload`. The primitives
        # already handle the resume case correctly:
        #  * `start_job` detects an already-inserted row from the original
        #    `harbor run --upload` and preserves it.
        #  * The streaming hook's `upload_single_trial` short-circuits on
        #    trials already in the DB (trial_exists check).
        #  * The post-run `upload_job` sweep lists existing trial ids and
        #    skips them, then finalizes only if archive_path is still NULL.
        if upload:
            await _setup_harbor_hub_streaming(
                job,
                public=public,
                share_orgs=requested_share_orgs,
                share_users=requested_share_users,
                confirm_non_member_orgs=confirm_non_member_orgs,
            )
        job_result = await job.run()

        # Streaming + post-run finalize must share an event loop — see the
        # long comment in `_run_harbor_hub_upload` for why.
        if upload:
            await _run_harbor_hub_upload(
                job.job_dir,
                public=public,
                share_orgs=requested_share_orgs,
                share_users=requested_share_users,
                confirm_non_member_orgs=confirm_non_member_orgs,
                yes=yes,
            )
        return job_result

    job_result = run_async(_run_job())

    # Print results tables
    print_job_results_tables(job_result)


@jobs_app.command()
def summarize(
    job_path: Annotated[Path, Argument(help="Path to job directory.")] = Path("."),
):
    """Summarize trial failures in a job using Claude Agent SDK."""
    console.print(
        "[red]Error: 'harbor jobs summarize' has been removed. "
        "Use 'harbor analyze <job-dir>' instead.[/red]"
    )
    raise SystemExit(1)


def _humanize_bytes(n: int) -> str:
    size: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@jobs_app.command()
def share(
    job_id: Annotated[str, Argument(help="Job ID (UUID) to share.")],
    share_org: Annotated[
        list[str] | None,
        Option("--share-org", help="Share the job with an organization. Repeatable."),
    ] = None,
    share_user: Annotated[
        list[str] | None,
        Option(
            "--share-user", help="Share the job with a GitHub username. Repeatable."
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "--yes",
            "-y",
            help="Confirm sharing with organizations you are not a member of.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        Option("--debug", help="Show extra details on failure.", hidden=True),
    ] = False,
) -> None:
    """Add org or user shares to an already-uploaded job."""
    from harbor.cli.job_sharing import (
        confirm_non_member_org_shares,
        format_share_summary,
        normalize_share_values,
    )
    from harbor.upload.db_client import UploadDB

    try:
        parsed_job_id = UUID(job_id)
    except ValueError:
        console.print("[red]Error:[/red] job_id must be a UUID.")
        raise SystemExit(1) from None

    requested_share_orgs = normalize_share_values(share_org)
    requested_share_users = normalize_share_values(share_user)
    if not requested_share_orgs and not requested_share_users:
        console.print("[red]Error:[/red] provide --share-org or --share-user.")
        raise SystemExit(1)

    async def _share() -> None:
        confirm_non_member_orgs = await confirm_non_member_org_shares(
            requested_share_orgs,
            yes=yes,
        )
        db = UploadDB()
        await db.get_user_id()
        result = await db.add_job_shares(
            job_id=parsed_job_id,
            org_names=requested_share_orgs,
            usernames=requested_share_users,
            confirm_non_member_orgs=confirm_non_member_orgs,
        )
        share_summary = format_share_summary(
            share_orgs=[
                org["name"]
                for org in result.get("orgs", [])
                if isinstance(org, dict) and org.get("name")
            ],
            share_users=[
                user.get("github_username") or user.get("id")
                for user in result.get("users", [])
                if isinstance(user, dict)
                and (user.get("github_username") or user.get("id"))
            ],
        )
        console.print(f"Updated job shares for {parsed_job_id}.")
        if share_summary:
            console.print(f"Shared with {share_summary}")

    try:
        run_async(_share())
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
        if debug:
            raise
        raise SystemExit(1) from None


@jobs_app.command()
def download(
    job_id: Annotated[str, Argument(help="Job ID (UUID) to download.")],
    output_dir: Annotated[
        Path,
        Option(
            "--output-dir",
            "-o",
            help="Directory in which to materialize the job_dir. Defaults to ./jobs.",
        ),
    ] = Path("jobs"),
    overwrite: Annotated[
        bool,
        Option("--overwrite", help="Replace an existing job_dir if present."),
    ] = False,
    debug: Annotated[
        bool,
        Option("--debug", help="Show extra details on failure.", hidden=True),
    ] = False,
) -> None:
    """Download a job (and all its trials) from the Harbor platform."""
    from uuid import UUID

    from rich.console import Console
    from typer import echo

    from harbor.cli.utils import run_async

    local_console = Console()

    try:
        job_uuid = UUID(job_id)
    except ValueError:
        echo(f"Error: {job_id!r} is not a valid UUID.")
        raise SystemExit(1) from None

    async def _download() -> None:
        from harbor.download.downloader import Downloader

        downloader = Downloader()

        # Friendly auth error — mirror upload's behaviour.
        try:
            await downloader.db.get_user_id()
        except RuntimeError as exc:
            echo(str(exc))
            raise SystemExit(1) from None

        # Pass the user's path through as-is (no `.resolve()`) so the
        # summary + suggested `harbor view` / `harbor analyze` commands
        # print in the same relative/absolute form the user typed. Default
        # `-o jobs` stays `jobs` instead of expanding to an unwieldy
        # cwd-prefixed absolute path.
        with local_console.status(f"[cyan]Downloading job {job_uuid}..."):
            result = await downloader.download_job(
                job_uuid, output_dir, overwrite=overwrite
            )

        echo(
            f"Downloaded {result.job_name} "
            f"({_humanize_bytes(result.archive_size_bytes)}) "
            f"in {result.download_time_sec:.2f}s → {result.output_dir}"
        )
        echo(f"Inspect locally by running `harbor view {result.output_dir.parent}`")
        echo(f"Analyze trajectories by running `harbor analyze {result.output_dir}`")

    try:
        run_async(_download())
    except SystemExit:
        raise
    except Exception as exc:
        echo(f"Error: {type(exc).__name__}: {exc}")
        if debug:
            raise
        raise SystemExit(1) from None


jobs_app.command()(start)
