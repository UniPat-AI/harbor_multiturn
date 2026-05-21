import asyncio
import contextlib
import copy
import json
import os
import shlex
import shutil
import traceback
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentMode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.trial.result import EnvironmentStateInfo, ExceptionInfo, TimingInfo
from harbor.models.verifier.result import VerifierResult
from harbor.trial.errors import AgentTimeoutError, VerifierTimeoutError
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial
from harbor.utils.env import resolve_env_vars
from harbor.utils.scripts import build_execution_command, needs_chmod, quote_shell_arg


class SingleStepTrial(Trial):
    """A trial with one instruction, one agent run, and one optional verifier."""

    def __init__(
        self,
        config: TrialConfig,
        *,
        _task: Task | None = None,
    ):
        if _task is not None and _task.has_steps:
            raise ValueError("SingleStepTrial requires a task without [[steps]].")
        super().__init__(config, _task=_task)
        self._are_artifacts_collected = False

    async def _run(self) -> None:
        if self.task.is_multiround:
            await self._execute_multiround()
            await self._collect_artifacts()
            await self._stop_agent_environment()
            return

        mode = resolve_task_verifier_mode(self.task.config)

        await self._run_agent()
        await self._upload_agent_logs()
        await self._collect_artifacts()

        if mode == VerifierEnvironmentMode.SEPARATE:
            await self._stop_agent_environment()

        await self._run_verifier()

        if mode == VerifierEnvironmentMode.SHARED:
            await self._stop_agent_environment()

    async def _recover_outputs(self) -> None:
        await self._sync_agent_output(self.result)
        await self._collect_artifacts()
        await self._stop_agent_environment()

    async def _collect_artifacts(self) -> None:
        if self._are_artifacts_collected:
            return

        await self._artifact_handler.download_artifacts(
            self.agent_environment,
            self.paths.artifacts_dir,
            source_artifacts_dir=self.agent_env_paths.artifacts_dir,
        )
        self._are_artifacts_collected = True

    async def _run_agent(self) -> None:
        try:
            await self._run_agent_phase(
                target=self.result,
                instruction=self.task.instruction,
                timeout_sec=self._agent_timeout_sec,
                user=self.task.config.agent.user,
            )
        except (AgentTimeoutError, NonZeroAgentExitCodeError) as exc:
            self._record_exception(exc)
        finally:
            await self._sync_agent_output(self.result)

    async def _run_verifier(self) -> None:
        if self.config.verifier.disable:
            return

        await self._emit(TrialEvent.VERIFICATION_START)
        self.result.verifier = TimingInfo(started_at=self._now())
        mode = resolve_task_verifier_mode(self.task.config)
        user = self.task.config.verifier.user
        try:
            if mode == VerifierEnvironmentMode.SEPARATE:
                self.result.verifier_result = await self._run_separate_verifier(
                    key="trial",
                    timeout_sec=self._verifier_timeout_sec,
                    artifacts_dir=self.paths.artifacts_dir,
                    user=user,
                )
            else:
                self.result.verifier_result = await self._run_shared_verifier(
                    timeout_sec=self._verifier_timeout_sec,
                    user=user,
                )
        except asyncio.TimeoutError as exc:
            raise VerifierTimeoutError(
                f"Verifier execution timed out after {self._verifier_timeout_sec} seconds"
            ) from exc
        finally:
            self.result.verifier.finished_at = self._now()

    async def _execute_multiround(self) -> None:
        """Run a multi-round task using one live environment across rounds."""
        start_round = self._verifier_config_value("multiround_start_round") or 1
        max_round = (
            self._verifier_config_value("multiround_max_round") or self.task.num_rounds
        )
        end_round = min(max_round, self.task.num_rounds)

        self._log_multiround_info(
            "Starting multi-round execution: rounds %s..%s (total task rounds: %s)",
            start_round,
            end_round,
            self.task.num_rounds,
        )
        resume_source = self._verifier_config_value("multiround_resume_source")
        if start_round > 1 and resume_source:
            resume_source_dir = Path(resume_source)
            self._log_multiround_info(
                "Resume lineage | source_trial=%s | source_dir=%s | completed_round=%s | snapshot_id=%s",
                resume_source_dir.name,
                resume_source_dir,
                start_round - 1,
                self._verifier_config_value("multiround_resume_state_snapshot_id")
                or "unknown",
            )

        round_results: list[dict[str, Any]] = []
        timeout_exception_info: ExceptionInfo | None = None
        timeout_traceback: str | None = None

        self.result.agent_execution = TimingInfo(started_at=self._now())
        self.result.verifier = TimingInfo(started_at=self._now())
        if self.result.agent_result is None:
            self.result.agent_result = AgentContext()

        if start_round > 1:
            resume_source = self._verifier_config_value("multiround_resume_source")
            inplace_resume = bool(
                self._verifier_config_value("multiround_resume_trial_name")
            )
            if not resume_source:
                self._log_multiround_info(
                    "No resume source provided; using default Oracle-prepared "
                    "late-round start for rounds 1..%s",
                    start_round - 1,
                )
            if resume_source:
                round_results.extend(
                    self._copy_resume_round_results(
                        Path(resume_source),
                        start_round,
                        preserve_status=inplace_resume,
                    )
                )

            if not (resume_source and self.agent_environment.restored_from_snapshot):
                try:
                    await self._fast_forward_rounds(start_round)
                except Exception as exc:
                    self.logger.error(
                        "Fast-forward failed at rounds 1..%s: %s",
                        start_round - 1,
                        exc,
                    )
                    self._aggregate_multiround_results(round_results)
                    raise

            if resume_source:
                await self._copy_resume_agent_state(Path(resume_source), start_round)
                self._log_multiround_info(
                    "--- Resume handoff complete; live execution begins at round %s ---",
                    start_round,
                )

        for round_num in range(start_round, end_round + 1):
            round_instruction = self.task.round_instruction(round_num)
            round_config = self.task.round_config(round_num)
            change_types = round_config.get("change_types", [])
            change_type = round_config.get("change_type", "unknown")
            change_types_display = ",".join(change_types) if change_types else "unknown"

            self._log_multiround_info(
                "=== Round %s/%s (change_types=%s) ===",
                round_num,
                self.task.num_rounds,
                change_types_display,
            )

            round_reward: float | int | None = None
            round_status: str | None = None
            should_stop = False

            await self._emit_if_configured(TrialEvent.AGENT_START)
            try:
                with self._agent_user_context():
                    await asyncio.wait_for(
                        self.agent.run_round(
                            instruction=round_instruction,
                            round_num=round_num,
                            environment=self.agent_environment,
                            context=self.result.agent_result,
                        ),
                        timeout=self._agent_timeout_sec,
                    )
            except asyncio.TimeoutError:
                timeout_error = AgentTimeoutError(
                    "Agent execution timed out after "
                    f"{self._agent_timeout_sec} seconds during round {round_num}"
                )
                timeout_exception_info = ExceptionInfo(
                    exception_type=type(timeout_error).__name__,
                    exception_message=str(timeout_error),
                    exception_traceback=traceback.format_exc(),
                    occurred_at=self._now(),
                )
                timeout_traceback = timeout_exception_info.exception_traceback
                self.logger.warning("Agent timed out during round %s", round_num)
                round_status = "agent_timeout"
                round_results.append(
                    self._round_result_payload(
                        round_num,
                        change_type,
                        change_types,
                        round_reward,
                        round_status,
                    )
                )
                should_stop = True
            except NonZeroAgentExitCodeError as exc:
                self._record_exception(exc)
                round_status = "agent_error"
                round_results.append(
                    self._round_result_payload(
                        round_num,
                        change_type,
                        change_types,
                        round_reward,
                        round_status,
                    )
                )
                should_stop = True
            else:
                if not self._verifier_config_value("disable", False):
                    await self._emit_if_configured(TrialEvent.VERIFICATION_START)
                    try:
                        round_reward = await self._verify_round(round_num)
                        self._log_multiround_info(
                            "Round %s reward: %s", round_num, round_reward
                        )
                        round_status = "completed"
                        round_results.append(
                            self._round_result_payload(
                                round_num,
                                change_type,
                                change_types,
                                round_reward,
                                round_status,
                            )
                        )
                        if round_reward != 1:
                            self._log_multiround_info(
                                "Round %s failed (reward=%s), stopping multi-round execution.",
                                round_num,
                                round_reward,
                            )
                            should_stop = True
                    except Exception as exc:
                        verification_traceback = traceback.format_exc()
                        self._log_multiround_warning(
                            "Verification did not produce a usable reward at round %s: %s. "
                            "Treating as round failure and stopping.",
                            round_num,
                            exc,
                        )
                        self.paths.exception_message_path.write_text(
                            verification_traceback
                        )
                        round_status = f"verification_error: {exc}"
                        round_results.append(
                            self._round_result_payload(
                                round_num,
                                change_type,
                                change_types,
                                round_reward,
                                round_status,
                            )
                        )
                        should_stop = True

                    self._backup_round_verifier_output(round_num)
                else:
                    round_status = "verification_disabled"
                    round_results.append(
                        self._round_result_payload(
                            round_num,
                            change_type,
                            change_types,
                            round_reward,
                            round_status,
                        )
                    )

            if round_status is not None:
                if not getattr(self, "_defer_multiround_state_snapshot", False):
                    await self._capture_environment_state_snapshot(
                        round_num=round_num,
                        round_reward=round_reward,
                        round_status=round_status,
                    )

            if should_stop:
                break

        self.result.agent_execution.finished_at = self._now()
        self.result.verifier.finished_at = self._now()

        await self._sync_agent_output_if_available()

        resume_source = self._verifier_config_value("multiround_resume_source")
        if start_round > 1 and resume_source:
            self._merge_resume_trajectory(
                Path(resume_source), start_round
            )

        self._aggregate_multiround_results(round_results)

        if timeout_exception_info is not None and self.result.exception_info is None:
            self.result.exception_info = timeout_exception_info
            if timeout_traceback is not None:
                self.paths.exception_message_path.write_text(timeout_traceback)

    @staticmethod
    def _round_result_payload(
        round_num: int,
        change_type: str,
        change_types: list[str],
        reward: float | int | None,
        status: str,
    ) -> dict[str, Any]:
        return {
            "round": round_num,
            "change_type": change_type,
            "change_types": change_types,
            "reward": reward,
            "status": status,
        }

    def _agent_environment_is_mounted(self) -> bool:
        capabilities = getattr(self.agent_environment, "capabilities", None)
        if capabilities is not None and hasattr(capabilities, "mounted"):
            return bool(capabilities.mounted)
        return bool(getattr(self.agent_environment, "is_mounted", False))

    def _agent_environment_os(self) -> TaskOS:
        return getattr(self.agent_environment, "os", TaskOS.LINUX)

    def _verifier_config_value(self, name: str, default: Any = None) -> Any:
        verifier_config = getattr(getattr(self, "config", None), "verifier", None)
        return getattr(verifier_config, name, default)

    def _trial_name(self) -> str:
        return getattr(getattr(self, "config", None), "trial_name", "trial")

    async def _emit_if_configured(self, event: TrialEvent) -> None:
        if not hasattr(self, "_hooks"):
            return
        config = getattr(self, "config", None)
        task = getattr(self, "task", None)
        if (
            config is None
            or task is None
            or not hasattr(config, "trial_name")
            or not hasattr(task, "name")
        ):
            return
        await self._emit(event)

    def _agent_user_context(self):
        if not hasattr(self.agent_environment, "with_default_user"):
            return contextlib.nullcontext()

        task_config = getattr(self.task, "config", None)
        agent_config = getattr(task_config, "agent", None)
        agent_user = getattr(agent_config, "user", None)
        return self.agent_environment.with_default_user(agent_user)

    async def _sync_agent_output_if_available(self) -> None:
        if (
            not hasattr(self, "_sync_agent_output")
            or (
                not hasattr(self.agent_environment, "capabilities")
                and not hasattr(self.agent_environment, "download_dir")
            )
        ):
            return
        await self._sync_agent_output(self.result)

    async def _fast_forward_rounds(self, up_to_round: int) -> None:
        """Apply oracle solve scripts for rounds 1..up_to_round-1."""
        task_os = self._agent_environment_os()
        env_paths = EnvironmentPaths.for_os(task_os)
        for ff_round in range(1, up_to_round):
            round_solution_dir = self.task.paths.round_solution_dir(ff_round)
            round_solve_path = self.task.paths.discovered_round_solve_path_for(
                ff_round, task_os
            )

            if round_solve_path is None:
                expected = self.task.paths.round_solve_path_for(
                    ff_round, task_os
                )
                raise FileNotFoundError(
                    f"Fast-forward failed: {expected} does not exist"
                )

            await self.agent_environment.upload_dir(
                source_dir=round_solution_dir,
                target_dir=str(env_paths.solution_dir),
            )

            solve_script_path = (
                env_paths.solution_dir
                / round_solve_path.relative_to(round_solution_dir).as_posix()
            )

            if needs_chmod(solve_script_path):
                await self.agent_environment.exec(
                    command=(
                        "chmod +x "
                        f"{quote_shell_arg(solve_script_path, task_os)}"
                    ),
                    user="root",
                )

            env = {"DEBIAN_FRONTEND": "noninteractive"}
            if self.task.config.solution.env:
                env.update(resolve_env_vars(self.task.config.solution.env))

            result = await self.agent_environment.exec(
                command=build_execution_command(
                    solve_script_path,
                    task_os=task_os,
                ),
                env=env,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Fast-forward solve for round {ff_round} exited with "
                    f"code {result.return_code}: {result.stderr or result.stdout}"
                )
            self.logger.info("Fast-forwarded round %s", ff_round)

    def _copy_resume_round_results(
        self,
        source_trial_dir: Path,
        up_to_round: int,
        *,
        preserve_status: bool = False,
    ) -> list[dict[str, Any]]:
        """Copy verifier outputs from an old trial for rounds 1..up_to_round-1."""
        old_verifier = source_trial_dir / "verifier"
        new_verifier = self.paths.verifier_dir
        new_verifier.mkdir(parents=True, exist_ok=True)

        for ff_round in range(1, up_to_round):
            for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
                src = old_verifier / f"round_{ff_round}_{name}"
                if src.exists():
                    shutil.copy2(src, new_verifier / f"round_{ff_round}_{name}")

        old_results_path = old_verifier / "multiround_results.json"
        if old_results_path.exists():
            all_old = json.loads(old_results_path.read_text())
            return [
                dict(rr) if preserve_status else {**rr, "status": "fast_forwarded"}
                for rr in all_old
                if isinstance(rr, dict) and rr.get("round", 0) < up_to_round
            ]
        return []

    def _load_resume_source_agent_name(self, source_trial_dir: Path) -> str | None:
        config_path = source_trial_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            source_trial_config = TrialConfig.model_validate_json(config_path.read_text())
        except Exception as exc:
            self.logger.warning(
                "Failed to read resume source config from %s: %s",
                config_path,
                exc,
            )
            return None

        return source_trial_config.agent.name

    def _resolve_resume_claude_sessions_dir(
        self, source_trial_dir: Path, start_round: int
    ) -> Path | None:
        completed_round = start_round - 1

        if completed_round < 1:
            return None

        for candidate_dir in self._iter_resume_lineage_trial_dirs(source_trial_dir):
            source_paths = TrialPaths(trial_dir=candidate_dir)
            round_sessions_dir = source_paths.agent_round_sessions_dir(completed_round)
            if round_sessions_dir.exists():
                return round_sessions_dir

            latest_sessions_dir = source_paths.agent_sessions_dir
            if not latest_sessions_dir.exists():
                continue

            latest_round = self._resolve_resume_source_latest_round(candidate_dir)
            if latest_round != completed_round:
                self.logger.warning(
                    "Requested Claude resume source %s at round %s, but the latest "
                    "available agent/session state corresponds to round %s; checking "
                    "parent resume lineage if present",
                    candidate_dir,
                    completed_round,
                    latest_round,
                )
                continue

            self.logger.warning(
                "Round-specific Claude session snapshot missing for resume source %s "
                "at round %s; falling back to top-level sessions %s because the source "
                "trial's latest round is also %s",
                candidate_dir,
                completed_round,
                latest_sessions_dir,
                latest_round,
            )
            return latest_sessions_dir
        return None

    def _resolve_resume_terminus_runtime_state_path(
        self, source_trial_dir: Path, start_round: int
    ) -> Path | None:
        completed_round = start_round - 1
        if completed_round < 1:
            return None

        for candidate_dir in self._iter_resume_lineage_trial_dirs(source_trial_dir):
            source_paths = TrialPaths(trial_dir=candidate_dir)
            round_state_path = source_paths.terminus_round_runtime_state_path(
                completed_round
            )
            if round_state_path.exists():
                return round_state_path
        return None

    async def _copy_resume_agent_state(
        self, source_trial_dir: Path, start_round: int
    ) -> None:
        """Copy agent runtime state needed to continue from a previous trial."""
        source_agent_dir = source_trial_dir / "agent"
        if not source_agent_dir.exists():
            return

        self._capture_pre_resume_trajectory_snapshot(source_trial_dir, start_round)

        source_agent_name = self._load_resume_source_agent_name(source_trial_dir)
        if (
            source_agent_name == AgentName.ORACLE.value
            and self.agent.name() != AgentName.ORACLE.value
        ):
            self._log_multiround_info(
                "Resume source %s was produced by oracle; skipping %s continuation "
                "state restore and starting round %s from Oracle-prepared workspace",
                source_trial_dir,
                self.agent.name(),
                start_round,
            )
            return

        if self.agent.name() == AgentName.CLAUDE_CODE.value:
            source_sessions = self._resolve_resume_claude_sessions_dir(
                source_trial_dir, start_round
            )
            if source_sessions is None:
                raise RuntimeError(
                    "No matching Claude session snapshot was found for resume source "
                    f"{source_trial_dir} at round {start_round - 1}"
                )

            self._capture_pre_resume_claude_sessions_snapshot(
                source_trial_dir,
                source_sessions,
                start_round,
            )

            target_sessions = self.paths.agent_sessions_dir
            if target_sessions.exists():
                shutil.rmtree(target_sessions, ignore_errors=True)
            target_sessions.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_sessions, target_sessions, symlinks=True)
            self._restore_claude_round_start_metadata(target_sessions)
            if not self._agent_environment_is_mounted():
                task_os = self._agent_environment_os()
                remote_sessions_dir = (
                    EnvironmentPaths.for_os(task_os).agent_dir
                    / "sessions"
                )
                await self.agent_environment.exec(
                    f"rm -rf {shlex.quote(remote_sessions_dir.as_posix())}"
                )
                await self.agent_environment.upload_dir(
                    source_dir=target_sessions,
                    target_dir=remote_sessions_dir.as_posix(),
                )
            self._log_multiround_info(
                "Copied Claude sessions for resume at round %s: %s -> %s",
                start_round,
                source_sessions,
                target_sessions,
            )
            return

        if self.agent.name() == AgentName.TERMINUS_2.value:
            source_state_path = self._resolve_resume_terminus_runtime_state_path(
                source_trial_dir, start_round
            )
            if source_state_path is None:
                raise RuntimeError(
                    "No matching Terminus-2 runtime snapshot was found for resume source "
                    f"{source_trial_dir} at round {start_round - 1}"
                )

            self.agent.restore_resume_state(source_state_path)
            self._log_multiround_info(
                "Restored Terminus-2 runtime state for resume at round %s: %s",
                start_round,
                source_state_path,
            )

    def _capture_pre_resume_trajectory_snapshot(
        self, source_trial_dir: Path, start_round: int
    ) -> None:
        completed_round = start_round - 1
        if completed_round < 1:
            return

        source_trajectory_path = source_trial_dir / "agent" / "trajectory.json"
        if not source_trajectory_path.exists():
            return

        try:
            trajectory = json.loads(source_trajectory_path.read_text())
            steps = trajectory.get("steps")
            if not isinstance(steps, list):
                return

            filtered_steps: list[dict[str, Any]] = []
            for step in steps:
                if not isinstance(step, dict):
                    continue

                extra = step.get("extra")
                round_num = extra.get("round") if isinstance(extra, dict) else None
                if isinstance(round_num, int) and round_num > completed_round:
                    continue

                filtered_steps.append(copy.deepcopy(step))

            if not filtered_steps:
                return

            for idx, step in enumerate(filtered_steps, start=1):
                step["step_id"] = idx

            trajectory["steps"] = filtered_steps
            existing_extra = trajectory.get("extra")
            if not isinstance(existing_extra, dict):
                existing_extra = {}
            trajectory["extra"] = {
                **existing_extra,
                "multiround_pre_resume_snapshot": self._build_resume_lineage_metadata(
                    source_trial_dir=source_trial_dir,
                    start_round=start_round,
                ),
            }
            target_path = self.paths.agent_pre_resume_trajectory_path(completed_round)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
            metadata_path = self.paths.agent_pre_resume_metadata_path(completed_round)
            metadata_path.write_text(
                json.dumps(
                    self._build_resume_lineage_metadata(
                        source_trial_dir=source_trial_dir,
                        start_round=start_round,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to capture pre-resume trajectory snapshot from %s: %s",
                source_trajectory_path,
                exc,
            )

    def _capture_pre_resume_claude_sessions_snapshot(
        self,
        source_trial_dir: Path,
        source_sessions_dir: Path,
        start_round: int,
    ) -> None:
        completed_round = start_round - 1
        if completed_round < 1:
            return

        source_paths = TrialPaths(trial_dir=source_trial_dir)
        source_kind = (
            "round_snapshot"
            if source_sessions_dir == source_paths.agent_round_sessions_dir(completed_round)
            else "latest_sessions_fallback"
        )

        target_dir = self.paths.agent_pre_resume_session_dir(completed_round)
        target_sessions_dir = self.paths.agent_pre_resume_session_files_dir(
            completed_round
        )

        try:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_sessions_dir, target_sessions_dir, symlinks=True)

            metadata = {
                "round": completed_round,
                "source_trial": str(source_trial_dir),
                "source_sessions_dir": str(source_sessions_dir),
                "source_kind": source_kind,
                "created_at": self._now().isoformat(),
            }
            self.paths.agent_pre_resume_session_metadata_path(completed_round).write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False)
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to capture pre-resume Claude sessions from %s: %s",
                source_sessions_dir,
                exc,
            )

    def _merge_resume_trajectory(
        self, source_trial_dir: Path, from_round: int
    ) -> None:
        backup_traj_path = source_trial_dir / "agent" / "trajectory.json"
        new_traj_path = self.paths.agent_dir / "trajectory.json"

        if not backup_traj_path.exists() or not new_traj_path.exists():
            self.logger.debug(
                "Skipping trajectory merge: backup or new trajectory.json missing"
            )
            return

        try:
            backup_data = json.loads(backup_traj_path.read_text())
            new_data = json.loads(new_traj_path.read_text())

            backup_steps = backup_data.get("steps", [])
            new_steps = new_data.get("steps", [])

            old_steps = [
                s
                for s in backup_steps
                if s.get("extra", {}).get("round") is not None
                and s["extra"]["round"] < from_round
            ]

            if not old_steps:
                self.logger.debug(
                    "No old-round steps found in backup trajectory, skipping merge"
                )
                return

            old_step_counts: dict[str, int] = {}
            for step in old_steps:
                fingerprint = self._trajectory_step_fingerprint(step, ignore_round=True)
                old_step_counts[fingerprint] = old_step_counts.get(fingerprint, 0) + 1

            filtered_new_steps: list[dict[str, Any]] = []
            removed_inherited_steps = 0
            for step in new_steps:
                step_round = self._trajectory_step_round(step)
                if step_round is not None and step_round >= from_round:
                    filtered_new_steps.append(step)
                    continue

                fingerprint = self._trajectory_step_fingerprint(step, ignore_round=True)
                if old_step_counts.get(fingerprint, 0) > 0:
                    old_step_counts[fingerprint] -= 1
                    removed_inherited_steps += 1
                    continue
                filtered_new_steps.append(step)

            merged_steps = old_steps + filtered_new_steps
            for i, step in enumerate(merged_steps):
                step["step_id"] = i + 1

            new_data["steps"] = merged_steps
            self._annotate_trajectory_resume_context(
                trajectory_data=new_data,
                source_trial_dir=source_trial_dir,
                start_round=from_round,
            )
            new_traj_path.write_text(json.dumps(new_data, indent=2, ensure_ascii=False))
            self._log_multiround_info(
                "Merged %s fast-forwarded steps into trajectory.json "
                "(removed %s inherited duplicates, total: %s steps)",
                len(old_steps),
                removed_inherited_steps,
                len(merged_steps),
            )
        except Exception as exc:
            self.logger.warning("Failed to merge resume trajectory: %s", exc)

    def _annotate_trajectory_resume_context(
        self,
        *,
        trajectory_data: dict[str, Any],
        source_trial_dir: Path,
        start_round: int,
    ) -> None:
        extra = trajectory_data.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            trajectory_data["extra"] = extra

        extra["multiround_resume"] = self._build_resume_lineage_metadata(
            source_trial_dir=source_trial_dir,
            start_round=start_round,
        )

        resume_note = (
            "Multiround resume: fast-forwarded rounds "
            f"1..{start_round - 1} from {source_trial_dir.name}; "
            f"live execution begins at round {start_round}."
        )
        notes = trajectory_data.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            trajectory_data["notes"] = resume_note
        elif resume_note not in notes:
            trajectory_data["notes"] = f"{resume_note}\n\n{notes}"

    @staticmethod
    def _trajectory_step_round(step: dict[str, Any]) -> int | None:
        extra = step.get("extra")
        round_num = extra.get("round") if isinstance(extra, dict) else None
        return round_num if isinstance(round_num, int) else None

    @staticmethod
    def _trajectory_step_fingerprint(
        step: dict[str, Any], *, ignore_round: bool = False
    ) -> str:
        normalized = copy.deepcopy(step)
        normalized.pop("step_id", None)

        extra = normalized.get("extra")
        if isinstance(extra, dict) and ignore_round:
            extra = dict(extra)
            extra.pop("round", None)
            if extra:
                normalized["extra"] = extra
            else:
                normalized.pop("extra", None)

        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    def _clear_current_round_verifier_outputs(self) -> None:
        verifier_dir = self.paths.verifier_dir
        for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
            (verifier_dir / name).unlink(missing_ok=True)

    def _write_verifier_exit_code(self, return_code: int) -> Path:
        exit_code_path = self.paths.verifier_dir / "test-exit-code.txt"
        exit_code_path.write_text(str(return_code))
        return exit_code_path

    def _build_verifier_failure_message(
        self,
        round_num: int,
        reason: str,
        *,
        include_exit_code: bool,
    ) -> str:
        diagnostic_files = ["test-stdout.txt", "test-stderr.txt"]
        if include_exit_code:
            diagnostic_files.append("test-exit-code.txt")
        diagnostic_files.append("verification-error.txt")
        diagnostics_dir = self.paths.verifier_dir
        diagnostics = f"{diagnostics_dir} ({', '.join(diagnostic_files)})"
        return f"{reason} See {diagnostics}."

    def _write_verification_error(self, message: str) -> Path:
        error_path = self.paths.verifier_dir / "verification-error.txt"
        error_path.write_text(f"{message}\n")
        return error_path

    def _remove_env_files_command(self, paths: list[str]) -> str:
        task_os = self._agent_environment_os()
        if task_os == TaskOS.WINDOWS:
            return " & ".join(
                f"if exist {quote_shell_arg(path, task_os)} "
                f"del /F /Q {quote_shell_arg(path, task_os)}"
                for path in paths
            )
        return "rm -f " + " ".join(
            quote_shell_arg(path, task_os) for path in paths
        )

    async def _verify_round(self, round_num: int) -> float | int:
        """Run verification for a specific round using round-specific tests."""
        task_os = self._agent_environment_os()
        env_paths = EnvironmentPaths.for_os(task_os)
        round_tests_dir = self.task.paths.round_tests_dir(round_num)
        if hasattr(self.task.paths, "discovered_round_test_path_for"):
            round_test_path = self.task.paths.discovered_round_test_path_for(
                round_num, task_os
            )
        else:
            round_test_path = None
        if round_test_path is None:
            if hasattr(self.task.paths, "round_test_path_for"):
                expected = self.task.paths.round_test_path_for(round_num, task_os)
            else:
                expected = self.task.paths.round_test_path(round_num)
            if Path(expected).exists():
                round_test_path = Path(expected)
            else:
                round_test_path = None
        if round_test_path is None:
            raise FileNotFoundError(f"No round test script found at {expected}")

        if hasattr(self.agent_environment, "reset_dirs"):
            await self.agent_environment.reset_dirs(
                remove_dirs=[env_paths.tests_dir],
                create_dirs=[env_paths.tests_dir],
                chmod_dirs=[env_paths.tests_dir],
            )
        else:
            await self.agent_environment.exec(
                command=(
                    f"rm -rf {quote_shell_arg(env_paths.tests_dir, task_os)} && "
                    f"mkdir -p {quote_shell_arg(env_paths.tests_dir, task_os)} && "
                    f"chmod -R u+rwX,go+rX {quote_shell_arg(env_paths.tests_dir, task_os)}"
                ),
                user="root",
            )
        await self.agent_environment.upload_dir(
            source_dir=round_tests_dir,
            target_dir=str(env_paths.tests_dir),
        )

        await self.agent_environment.exec(
            self._remove_env_files_command(
                [
                    str(env_paths.reward_text_path),
                    str(env_paths.reward_json_path),
                ]
            ),
            user="root" if task_os != TaskOS.WINDOWS else None,
        )

        self._clear_current_round_verifier_outputs()
        self.paths.test_stdout_path.touch()
        self.paths.test_stderr_path.touch()

        test_script_path = (
            env_paths.tests_dir / round_test_path.relative_to(round_tests_dir).as_posix()
        )
        test_stdout_path = (
            env_paths.verifier_dir
            / self.paths.test_stdout_path.relative_to(self.paths.verifier_dir).as_posix()
        )

        if needs_chmod(test_script_path):
            await self.agent_environment.exec(
                command=(
                    "chmod +x "
                    f"{quote_shell_arg(test_script_path, task_os)}"
                ),
                user="root",
            )

        env = None
        if self.task.config.verifier.env:
            env = resolve_env_vars(self.task.config.verifier.env)

        try:
            exec_result = await asyncio.wait_for(
                self.agent_environment.exec(
                    command=build_execution_command(
                        test_script_path,
                        stdout_path=test_stdout_path,
                        task_os=task_os,
                    ),
                    env=env,
                ),
                timeout=self._verifier_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            if not self._agent_environment_is_mounted():
                await self.agent_environment.download_dir(
                    source_dir=str(env_paths.verifier_dir),
                    target_dir=self.paths.verifier_dir,
                )
            message = self._build_verifier_failure_message(
                round_num,
                (
                    "Verifier timed out after "
                    f"{self._verifier_timeout_sec} seconds during round {round_num}"
                ),
                include_exit_code=False,
            )
            self._write_verification_error(message)
            raise VerifierTimeoutError(message) from exc

        if not self._agent_environment_is_mounted():
            await self.agent_environment.download_dir(
                source_dir=str(env_paths.verifier_dir),
                target_dir=self.paths.verifier_dir,
            )

        self._write_verifier_exit_code(exec_result.return_code)
        if exec_result.return_code != 0:
            message = self._build_verifier_failure_message(
                round_num,
                f"Verifier exited with code {exec_result.return_code} during round {round_num}",
                include_exit_code=True,
            )
            self._write_verification_error(message)
            raise RuntimeError(message)

        if self.paths.reward_text_path.exists():
            reward_text = self.paths.reward_text_path.read_text().strip()
            if not reward_text:
                message = self._build_verifier_failure_message(
                    round_num,
                    f"Reward file is empty after round {round_num}",
                    include_exit_code=True,
                )
                self._write_verification_error(message)
                raise ValueError(message)
            return float(reward_text)

        if self.paths.reward_json_path.exists():
            reward_data = json.loads(self.paths.reward_json_path.read_text())
            return reward_data.get("reward", 0)

        message = self._build_verifier_failure_message(
            round_num,
            f"No reward file found after round {round_num}",
            include_exit_code=True,
        )
        self._write_verification_error(message)
        raise FileNotFoundError(message)

    def _backup_round_verifier_output(self, round_num: int) -> None:
        verifier_dir = self.paths.verifier_dir
        for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
            src = verifier_dir / name
            if src.exists():
                shutil.copy2(src, verifier_dir / f"round_{round_num}_{name}")

    def _aggregate_multiround_results(
        self, round_results: list[dict[str, Any]]
    ) -> None:
        rewards: dict[str, float | int] = {}
        round_results_by_round = {
            rr["round"]: rr
            for rr in round_results
            if isinstance(rr, dict) and isinstance(rr.get("round"), int)
        }

        start_round = self._verifier_config_value("multiround_start_round") or 1
        configured_max_round = self._verifier_config_value("multiround_max_round")
        task_num_rounds = self.task.num_rounds
        if not isinstance(task_num_rounds, int) or task_num_rounds < 1:
            raise ValueError(
                "Multi-round aggregation requires task.num_rounds to be available"
            )
        default_aggregate_end_round = min(
            configured_max_round or task_num_rounds,
            task_num_rounds,
        )
        aggregate_start_round = (
            1
            if self._verifier_config_value("multiround_resume_trial_name")
            else start_round
        )
        aggregate_start_round = (
            self._verifier_config_value("multiround_aggregate_start_round")
            or aggregate_start_round
        )
        aggregate_end_round = (
            self._verifier_config_value("multiround_aggregate_end_round")
            or default_aggregate_end_round
        )

        if aggregate_start_round > aggregate_end_round:
            raise ValueError(
                "Multi-round aggregation requires aggregate_start_round <= "
                "aggregate_end_round"
            )

        aggregate_rewards: list[float | int] = []
        for round_num in range(aggregate_start_round, aggregate_end_round + 1):
            round_result = round_results_by_round.get(round_num)
            round_reward = (
                round_result["reward"]
                if round_result is not None and round_result.get("reward") is not None
                else 0
            )
            rewards[f"round_{round_num}"] = round_reward
            aggregate_rewards.append(round_reward)

        rewards["reward"] = (
            sum(aggregate_rewards) / len(aggregate_rewards) if aggregate_rewards else 0
        )

        self.result.verifier_result = VerifierResult(
            rewards=rewards,
            aggregate_window_start=aggregate_start_round,
            aggregate_window_end=aggregate_end_round,
        )

        round_results_path = self.paths.verifier_dir / "multiround_results.json"
        round_results_path.write_text(json.dumps(round_results, indent=2))

    def _round_result_for_state_snapshot(
        self, round_num: int
    ) -> tuple[float | int | None, str]:
        round_results_path = self.paths.verifier_dir / "multiround_results.json"
        if round_results_path.exists():
            try:
                round_results = json.loads(round_results_path.read_text())
            except json.JSONDecodeError:
                round_results = []
            if isinstance(round_results, list):
                for round_result in round_results:
                    if (
                        isinstance(round_result, dict)
                        and round_result.get("round") == round_num
                    ):
                        status = round_result.get("status")
                        return (
                            round_result.get("reward"),
                            status if isinstance(status, str) else "unknown",
                        )

        round_reward = None
        if (
            self.result.verifier_result is not None
            and self.result.verifier_result.rewards is not None
        ):
            round_reward = self.result.verifier_result.rewards.get(f"round_{round_num}")
        return round_reward, "completed"

    async def capture_multiround_state_snapshot(
        self,
        *,
        round_num: int,
        restart_environment: bool = True,
    ) -> None:
        round_reward, round_status = self._round_result_for_state_snapshot(round_num)
        await self._capture_environment_state_snapshot(
            round_num=round_num,
            round_reward=round_reward,
            round_status=round_status,
            restart_environment=restart_environment,
        )
        self.paths.result_path.write_text(self.result.model_dump_json(indent=4))

    def _should_capture_state_snapshot(
        self,
        *,
        round_reward: float | int | None,
        round_status: str,
    ) -> bool:
        if not self.task.is_multiround:
            return False
        policy = self._verifier_config_value("multiround_state_cache_policy", "off")
        if policy == "off":
            return False
        if policy == "all":
            return round_status != "agent_timeout"
        if policy == "success":
            if round_status != "completed" or round_reward is None:
                return False
            try:
                return float(round_reward) == 1.0
            except (TypeError, ValueError):
                return False
        return False

    def _write_latest_state_snapshot_alias(
        self, round_num: int, snapshot_payload: dict[str, Any]
    ) -> str | None:
        latest_snapshot_path = self.paths.state_snapshot_path
        latest_archive_path = self.paths.state_image_archive_path
        round_archive_path = self.paths.round_state_image_archive_path(round_num)

        latest_payload = dict(snapshot_payload)
        latest_archive_ref = snapshot_payload.get("archive_path")
        if round_archive_path.exists():
            latest_archive_ref = str(latest_archive_path.expanduser().absolute())
        latest_payload["archive_path"] = latest_archive_ref
        latest_snapshot_path.write_text(json.dumps(latest_payload, indent=2))

        if latest_archive_path.exists() or latest_archive_path.is_symlink():
            latest_archive_path.unlink()

        if not round_archive_path.exists():
            return latest_archive_ref

        try:
            latest_archive_path.symlink_to(
                round_archive_path.relative_to(latest_archive_path.parent)
            )
        except OSError:
            shutil.copy2(round_archive_path, latest_archive_path)
        return latest_archive_ref

    async def _capture_environment_state_snapshot(
        self,
        *,
        round_num: int,
        round_reward: float | int | None,
        round_status: str,
        restart_environment: bool = True,
    ) -> None:
        if not self._should_capture_state_snapshot(
            round_reward=round_reward,
            round_status=round_status,
        ):
            return

        round_state_dir = self.paths.round_state_dir(round_num)
        round_state_dir.mkdir(parents=True, exist_ok=True)

        trial_name = self._trial_name()
        snapshot_id = f"{trial_name}__round-{round_num}__state"
        await self._capture_terminus_runtime_snapshot(round_num)
        await self._capture_claude_session_snapshot(round_num)
        snapshot_data = await self.agent_environment.capture_state_snapshot(
            snapshot_id=snapshot_id,
            archive_path=self.paths.round_state_image_archive_path(round_num),
            restart_container=restart_environment,
        )

        if snapshot_data is None:
            raise RuntimeError(
                "Failed to capture environment snapshot for "
                f"{trial_name} round {round_num}"
            )

        parent_snapshot_id = getattr(self, "_latest_snapshot_parent_id", None)
        snapshot_payload = {
            "snapshot_id": snapshot_data.get("snapshot_id"),
            "image_ref": snapshot_data.get("image_ref"),
            "image_tag": snapshot_data.get("image_tag"),
            "archive_path": snapshot_data.get("archive_path"),
            "parent_snapshot_id": parent_snapshot_id,
            "source_trial": self._verifier_config_value("multiround_resume_source"),
            "trial_name": trial_name,
            "round": round_num,
            "reward": round_reward,
            "status": round_status,
            "created_at": self._now().isoformat(),
        }
        for key in (
            "provider",
            "provider_state_mode",
            "daytona_sandbox_id",
            "daytona_sandbox_name",
            "daytona_snapshot_name",
        ):
            if key in snapshot_data:
                snapshot_payload[key] = snapshot_data.get(key)
        if (
            snapshot_payload.get("provider") == "daytona"
            and snapshot_payload.get("provider_state_mode") == "pause_fork"
            and snapshot_payload.get("archive_path")
        ):
            snapshot_payload["provider_fallback_state_mode"] = "archive"
        self.paths.round_state_snapshot_path(round_num).write_text(
            json.dumps(snapshot_payload, indent=2)
        )
        latest_archive_ref = self._write_latest_state_snapshot_alias(
            round_num, snapshot_payload
        )
        self._latest_snapshot_parent_id = snapshot_payload.get("snapshot_id")

        self.result.environment_state = EnvironmentStateInfo(
            snapshot_id=snapshot_payload.get("snapshot_id"),
            image_ref=snapshot_payload.get("image_tag")
            or snapshot_payload.get("image_ref"),
            image_archive=latest_archive_ref,
            parent_snapshot_id=snapshot_payload.get("parent_snapshot_id"),
            source=snapshot_payload.get("source_trial"),
        )

    async def _capture_claude_session_snapshot(self, round_num: int) -> None:
        if self.agent.name() != AgentName.CLAUDE_CODE.value:
            return

        live_sessions_dir = self.paths.agent_sessions_dir
        round_sessions_dir = self.paths.agent_round_sessions_dir(round_num)
        env_paths = EnvironmentPaths.for_os(self._agent_environment_os())
        remote_sessions_dir = (env_paths.agent_dir / "sessions").as_posix()

        if not self._agent_environment_is_mounted():
            if live_sessions_dir.exists():
                shutil.rmtree(live_sessions_dir, ignore_errors=True)
            try:
                await self.agent_environment.download_dir(
                    source_dir=remote_sessions_dir,
                    target_dir=live_sessions_dir,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download Claude sessions for "
                    f"{self._trial_name()} round {round_num}"
                ) from exc
        else:
            try:
                await self.agent_environment.exec(
                    " ".join(
                        [
                            "if",
                            "[",
                            "-e",
                            shlex.quote(remote_sessions_dir),
                            "];",
                            "then",
                            "chown",
                            "-R",
                            f"{os.getuid()}:{os.getgid()}",
                            shlex.quote(remote_sessions_dir),
                            "&&",
                            "chmod",
                            "-R",
                            "u+rwX,go+rX",
                            shlex.quote(remote_sessions_dir),
                            ";",
                            "fi",
                        ]
                    ),
                    user="root",
                )
            except Exception:
                self.logger.warning(
                    "Failed to fix permissions on Claude sessions inside container: %s",
                    remote_sessions_dir,
                )

        if not live_sessions_dir.exists():
            raise RuntimeError(
                "Claude sessions directory was not found for "
                f"{self._trial_name()} round {round_num}"
            )

        self._write_claude_round_start_metadata(live_sessions_dir)

        if round_sessions_dir.exists():
            shutil.rmtree(round_sessions_dir, ignore_errors=True)
        round_sessions_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(live_sessions_dir, round_sessions_dir, symlinks=True)

    async def _capture_terminus_runtime_snapshot(self, round_num: int) -> None:
        if self.agent.name() != AgentName.TERMINUS_2.value:
            return

        snapshot_path = self.paths.terminus_round_runtime_state_path(round_num)
        try:
            await self.agent.capture_resume_state(snapshot_path, round_num)
        except Exception as exc:
            raise RuntimeError(
                "Failed to capture Terminus-2 runtime snapshot for "
                f"{self._trial_name()} round {round_num}"
            ) from exc

    def _write_claude_round_start_metadata(self, sessions_dir: Path) -> None:
        if self.agent.name() != AgentName.CLAUDE_CODE.value:
            return
        round_starts = getattr(self.agent, "_round_starts", None)
        if round_starts is None:
            return

        metadata_path = sessions_dir / self._CLAUDE_ROUND_STARTS_FILENAME
        payload = [
            {"round": round_num, "started_at": started_at}
            for round_num, started_at in round_starts
        ]
        metadata_path.write_text(json.dumps(payload, indent=2))

    def _restore_claude_round_start_metadata(self, sessions_dir: Path) -> None:
        if self.agent.name() != AgentName.CLAUDE_CODE.value:
            return

        metadata_path = sessions_dir / self._CLAUDE_ROUND_STARTS_FILENAME
        if not metadata_path.exists():
            return

        payload = json.loads(metadata_path.read_text())
        round_starts: list[tuple[int, str]] = []
        for item in payload:
            if isinstance(item, dict):
                round_num = item.get("round")
                started_at = item.get("started_at")
            elif isinstance(item, list | tuple) and len(item) == 2:
                round_num, started_at = item
            else:
                continue

            if isinstance(round_num, int) and isinstance(started_at, str):
                round_starts.append((round_num, started_at))

        if round_starts:
            setattr(self.agent, "_round_starts", round_starts)
