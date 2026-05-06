import asyncio
import copy
import json
import logging
import os
import shlex
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.task.task import Task
from harbor.models.trial.config import ArtifactConfig, TrialConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.trial.result import (
    EnvironmentStateInfo,
    ExceptionInfo,
    TimingInfo,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult
from harbor.tasks.client import TaskClient
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.utils.logger import logger
from harbor.utils.env import resolve_env_vars
from harbor.verifier.verifier import Verifier


class AgentSetupTimeoutError(asyncio.TimeoutError):
    pass


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class VerifierTimeoutError(asyncio.TimeoutError):
    pass


class EnvironmentStartTimeoutError(asyncio.TimeoutError):
    pass


TrialHookCallback = Callable[[TrialHookEvent], Awaitable[None]]


class Trial:
    """
    Runs a trial of a given agent on an environment.

    1. Initializes the environment.
    2. Runs the agent on the environment.
    3. Verifies the results.
    4. Saves the results.
    5. Cleans up the environment.
    6. Uploads the results.
    """

    _AGENT_SETUP_TIMEOUT_SEC = 360
    _CLAUDE_ROUND_STARTS_FILENAME = ".harbor-round-starts.json"
    _MULTIROUND_VERIFIER_ARTIFACT_NAMES = (
        "test-stdout.txt",
        "test-stderr.txt",
        "test-exit-code.txt",
        "verification-error.txt",
        "reward.txt",
        "reward.json",
    )

    def __init__(self, config: TrialConfig):
        self.config = config
        self.job_id = config.job_id
        self._are_agent_logs_downloaded = False

        self._hooks: dict[TrialEvent, list[TrialHookCallback]] = {
            event: [] for event in TrialEvent
        }

        self._task = self._load_task()
        self._trial_paths = TrialPaths(trial_dir=self.trial_dir)
        self._trial_paths.mkdir()

        self._log_handler: logging.Handler | None = None
        self._init_logger()

        extra_kwargs = {}
        if config.agent.name == AgentName.ORACLE.value:
            extra_kwargs = {
                "task_dir": self._task._task_dir,
                "trial_paths": self._trial_paths,
            }
        if self._task.config.environment.mcp_servers:
            extra_kwargs["mcp_servers"] = self._task.config.environment.mcp_servers
        if self._task.config.environment.skills_dir:
            extra_kwargs["skills_dir"] = self._task.config.environment.skills_dir

        self._agent = AgentFactory.create_agent_from_config(
            config.agent,
            logs_dir=self._trial_paths.agent_dir,
            logger=self._logger,
            **extra_kwargs,
        )

        (
            self._resume_state_image_ref,
            self._resume_state_archive_path,
            self._resume_state_snapshot_id,
        ) = self._resolve_resume_state()
        self._latest_snapshot_parent_id = self._resume_state_snapshot_id

        self._environment = EnvironmentFactory.create_environment_from_config(
            config=config.environment,
            environment_dir=self._task.paths.environment_dir,
            environment_name=self._task.name,
            session_id=self.config.trial_name,
            trial_paths=self._trial_paths,
            task_env_config=self._task.config.environment,
            logger=self._logger,
            resume_state_image_ref=self._resume_state_image_ref,
            resume_state_archive_path=self._resume_state_archive_path,
        )
        self._environment_supports_state_snapshots = (
            type(self._environment).capture_state_snapshot
            is not BaseEnvironment.capture_state_snapshot
        )

        self._agent_timeout_sec = min(
            config.agent.override_timeout_sec or self._task.config.agent.timeout_sec,
            config.agent.max_timeout_sec or float("inf"),
        ) * (
            config.agent_timeout_multiplier
            if config.agent_timeout_multiplier is not None
            else config.timeout_multiplier
        )

        self._verifier_timeout_sec = min(
            config.verifier.override_timeout_sec
            or self._task.config.verifier.timeout_sec,
            config.verifier.max_timeout_sec or float("inf"),
        ) * (
            config.verifier_timeout_multiplier
            if config.verifier_timeout_multiplier is not None
            else config.timeout_multiplier
        )

        self._agent_setup_timeout_sec = (
            config.agent.override_setup_timeout_sec
            if config.agent.override_setup_timeout_sec is not None
            else self._AGENT_SETUP_TIMEOUT_SEC
        ) * (
            config.agent_setup_timeout_multiplier
            if config.agent_setup_timeout_multiplier is not None
            else self.config.timeout_multiplier
        )

        self._environment_build_timeout_sec = (
            self._task.config.environment.build_timeout_sec
            * (
                config.environment_build_timeout_multiplier
                if config.environment_build_timeout_multiplier is not None
                else self.config.timeout_multiplier
            )
        )

        self._result: TrialResult | None = None

    @property
    def result(self) -> TrialResult:
        if self._result is None:
            raise RuntimeError("Trial result accessed before initialization")
        return self._result

    def _init_logger(self):
        self._logger = logger.getChild(f"{__name__}.{self.config.trial_name}")
        file_handler = logging.FileHandler(self._trial_paths.log_path)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)
        self._log_handler = file_handler

    @staticmethod
    def _is_plain_quiet_progress_enabled() -> bool:
        return os.getenv("HARBOR_PLAIN_QUIET_PROGRESS", "false").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    def _multiround_trial_prefix(self) -> str:
        trial_name = getattr(self.config, "trial_name", "unknown-trial")
        return f"[trial={trial_name}]"

    def _log_multiround_info(self, message: str, *args: Any) -> None:
        prefixed_message = f"{self._multiround_trial_prefix()} {message}"
        if self._is_plain_quiet_progress_enabled():
            self._logger.debug(prefixed_message, *args)
        else:
            self._logger.info(prefixed_message, *args)

    def _log_multiround_warning(self, message: str, *args: Any) -> None:
        prefixed_message = f"{self._multiround_trial_prefix()} {message}"
        self._logger.warning(prefixed_message, *args)

    def _build_resume_lineage_metadata(
        self,
        *,
        source_trial_dir: Path,
        start_round: int,
    ) -> dict[str, Any]:
        completed_round = start_round - 1
        verifier_config = getattr(self.config, "verifier", None)
        return {
            "resume_source_trial": source_trial_dir.name,
            "resume_source_dir": str(source_trial_dir),
            "resume_completed_round": completed_round,
            "resume_into_round": start_round,
            "resume_state_snapshot_id": getattr(
                verifier_config, "multiround_resume_state_snapshot_id", None
            ),
            "resume_state_image": getattr(
                verifier_config, "multiround_resume_state_image", None
            ),
        }

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

    def _close_logger_handler(self) -> None:
        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def add_hook(self, event: TrialEvent, hook: TrialHookCallback) -> None:
        """Add an async hook to be called when the specified event occurs."""
        self._hooks[event].append(hook)

    async def _invoke_hooks(self, event: TrialEvent) -> None:
        """Invoke all hooks registered for the given event."""
        hook_event = TrialHookEvent(
            event=event,
            trial_id=self.config.trial_name,
            task_name=self._task.name,
            config=self.config,
            result=self._result,
        )
        for hook in self._hooks[event]:
            await hook(hook_event)

    def _load_task(self) -> Task:
        if self.config.task.is_git_task():
            client = TaskClient()
            task_id = self.config.task.get_task_id()

            task_dir = client.download_tasks(
                task_ids=[task_id],
                overwrite=self.config.task.overwrite,
                output_dir=self.config.task.download_dir,
            )[0]

            return Task(task_dir=task_dir)
        else:
            return Task(task_dir=self.config.task.path)

    @property
    def trial_dir(self) -> Path:
        return self.config.trials_dir / self.config.trial_name

    def _resolve_resume_snapshot_metadata_path(
        self, resume_source: Path
    ) -> Path | None:
        source_paths = TrialPaths(trial_dir=resume_source)
        start_round = self.config.verifier.multiround_start_round or 1
        completed_round = start_round - 1

        if completed_round >= 1:
            round_snapshot_path = source_paths.round_state_snapshot_path(completed_round)
            if round_snapshot_path.exists():
                return round_snapshot_path

            latest_snapshot_path = source_paths.state_snapshot_path
            if latest_snapshot_path.exists():
                try:
                    latest_snapshot_payload = json.loads(latest_snapshot_path.read_text())
                except Exception:
                    latest_snapshot_payload = {}

                latest_round = latest_snapshot_payload.get("round")
                if latest_round not in (None, completed_round):
                    self._logger.warning(
                        "Requested resume source %s at round %s, but top-level latest "
                        "snapshot %s points to round %s; refusing fallback to avoid "
                        "restoring the wrong state",
                        resume_source,
                        completed_round,
                        latest_snapshot_path,
                        latest_round,
                    )
                    return None

                self._logger.warning(
                    "Round-specific snapshot metadata missing for resume source %s "
                    "at round %s; falling back to latest snapshot metadata %s",
                    resume_source,
                    completed_round,
                    latest_snapshot_path,
                )
                return latest_snapshot_path
            return None

        latest_snapshot_path = source_paths.state_snapshot_path
        if latest_snapshot_path.exists():
            return latest_snapshot_path
        return None

    def _resolve_resume_source_latest_round(self, source_trial_dir: Path) -> int | None:
        source_paths = TrialPaths(trial_dir=source_trial_dir)
        latest_snapshot_path = source_paths.state_snapshot_path
        if latest_snapshot_path.exists():
            try:
                latest_snapshot_payload = json.loads(latest_snapshot_path.read_text())
            except Exception:
                latest_snapshot_payload = {}

            latest_round = latest_snapshot_payload.get("round")
            if isinstance(latest_round, int):
                return latest_round

        round_results_path = source_paths.verifier_dir / "multiround_results.json"
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

    def _resolve_resume_claude_sessions_dir(
        self, source_trial_dir: Path, start_round: int
    ) -> Path | None:
        source_paths = TrialPaths(trial_dir=source_trial_dir)
        completed_round = start_round - 1

        if completed_round < 1:
            return None

        round_sessions_dir = source_paths.agent_round_sessions_dir(completed_round)
        if round_sessions_dir.exists():
            return round_sessions_dir

        latest_sessions_dir = source_paths.agent_sessions_dir
        if not latest_sessions_dir.exists():
            return None

        latest_round = self._resolve_resume_source_latest_round(source_trial_dir)
        if latest_round != completed_round:
            self._logger.warning(
                "Requested Claude resume source %s at round %s, but the latest "
                "available agent/session state corresponds to round %s; refusing "
                "fallback to top-level sessions to avoid restoring a future round",
                source_trial_dir,
                completed_round,
                latest_round,
            )
            return None

        self._logger.warning(
            "Round-specific Claude session snapshot missing for resume source %s "
            "at round %s; falling back to top-level sessions %s because the source "
            "trial's latest round is also %s",
            source_trial_dir,
            completed_round,
            latest_sessions_dir,
            latest_round,
        )
        return latest_sessions_dir

    def _resolve_resume_terminus_runtime_state_path(
        self, source_trial_dir: Path, start_round: int
    ) -> Path | None:
        source_paths = TrialPaths(trial_dir=source_trial_dir)
        completed_round = start_round - 1
        if completed_round < 1:
            return None

        round_state_path = source_paths.terminus_round_runtime_state_path(
            completed_round
        )
        if round_state_path.exists():
            return round_state_path
        return None

    def _resolve_resume_state(self) -> tuple[str | None, str | None, str | None]:
        """Resolve resume snapshot references from config or resume source trial dir."""
        image_ref = self.config.verifier.multiround_resume_state_image
        archive_path = self.config.verifier.multiround_resume_state_archive
        snapshot_id = self.config.verifier.multiround_resume_state_snapshot_id

        resume_source = self.config.verifier.multiround_resume_source
        if resume_source is None:
            return image_ref, archive_path, snapshot_id

        snapshot_meta_path = self._resolve_resume_snapshot_metadata_path(
            Path(resume_source)
        )
        if snapshot_meta_path is None:
            if not image_ref:
                raise RuntimeError(
                    "No matching resume snapshot metadata was found for resume source: "
                    f"{resume_source}"
                )
            return image_ref, archive_path, snapshot_id

        try:
            snapshot_meta = json.loads(snapshot_meta_path.read_text())
        except Exception as e:
            self._logger.warning(
                f"Failed to read resume snapshot metadata from {snapshot_meta_path}: {e}"
            )
            return image_ref, archive_path, snapshot_id

        image_ref = image_ref or snapshot_meta.get("image_tag") or snapshot_meta.get(
            "image_ref"
        )
        archive_path = archive_path or snapshot_meta.get("archive_path")
        snapshot_id = snapshot_id or snapshot_meta.get("snapshot_id")

        if archive_path:
            archive_candidate = Path(archive_path).expanduser()
            if archive_candidate.is_absolute():
                archive_path = str(archive_candidate)
            elif archive_candidate.exists():
                archive_path = str(archive_candidate.resolve())
            else:
                metadata_relative_candidate = (
                    snapshot_meta_path.parent / archive_candidate
                )
                if metadata_relative_candidate.exists():
                    archive_path = str(metadata_relative_candidate.resolve())
                else:
                    source_relative_candidate = Path(resume_source) / archive_candidate
                    if source_relative_candidate.exists():
                        archive_path = str(source_relative_candidate.resolve())

        if not image_ref:
            raise RuntimeError(
                "No resume snapshot image metadata was found for resume source: "
                f"{resume_source}"
            )

        return image_ref, archive_path, snapshot_id

    async def _setup_environment(self) -> None:
        await self._invoke_hooks(TrialEvent.ENVIRONMENT_START)

        self.result.environment_setup = TimingInfo(
            started_at=datetime.now(timezone.utc)
        )

        try:
            await self._start_environment_with_retry()
        finally:
            self.result.environment_setup.finished_at = datetime.now(timezone.utc)

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EnvironmentStartTimeoutError),
    )
    async def _start_environment_with_retry(self) -> None:
        try:
            await asyncio.wait_for(
                self._environment.start(
                    force_build=self.config.environment.force_build
                ),
                timeout=self._environment_build_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise EnvironmentStartTimeoutError(
                f"Environment start timed out after "
                f"{self._environment_build_timeout_sec} seconds"
            ) from e

    async def _setup_agent(self) -> None:
        self.result.agent_setup = TimingInfo(started_at=datetime.now(timezone.utc))
        try:
            await asyncio.wait_for(
                self._agent.setup(environment=self._environment),
                timeout=self._agent_setup_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise AgentSetupTimeoutError(
                f"Agent setup timed out after {self._agent_setup_timeout_sec} seconds"
            ) from e
        finally:
            self.result.agent_setup.finished_at = datetime.now(timezone.utc)

    async def _execute_agent(self) -> None:
        await self._invoke_hooks(TrialEvent.AGENT_START)

        self.result.agent_execution = TimingInfo(started_at=datetime.now(timezone.utc))

        try:
            self.result.agent_result = AgentContext()

            await asyncio.wait_for(
                self._agent.run(
                    instruction=self._task.instruction,
                    environment=self._environment,
                    context=self.result.agent_result,
                ),
                timeout=self._agent_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise AgentTimeoutError(
                f"Agent execution timed out after {self._agent_timeout_sec} seconds"
            ) from e
        finally:
            self.result.agent_execution.finished_at = datetime.now(timezone.utc)

    async def _run_verification(self) -> None:
        await self._invoke_hooks(TrialEvent.VERIFICATION_START)

        self.result.verifier = TimingInfo(started_at=datetime.now(timezone.utc))

        try:
            await self._verify_with_retry()
        finally:
            self.result.verifier.finished_at = datetime.now(timezone.utc)

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(VerifierTimeoutError),
    )
    async def _verify_with_retry(self) -> None:
        try:
            verifier = Verifier(
                task=self._task,
                trial_paths=self._trial_paths,
                environment=self._environment,
            )

            self.result.verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._verifier_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            raise VerifierTimeoutError(
                f"Verifier execution timed out after "
                f"{self._verifier_timeout_sec} seconds"
            ) from e

    async def _cleanup_and_finalize(self) -> None:
        try:
            await self._environment.stop(delete=self.config.environment.delete)
        except Exception as e:
            logger.warning(
                f"Warning: Environment cleanup failed for {self.config.trial_name}: {e}"
            )
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)

        self.result.finished_at = datetime.now(timezone.utc)

        self._trial_paths.result_path.write_text(self.result.model_dump_json(indent=4))

        await self._invoke_hooks(TrialEvent.END)

    async def _maybe_download_logs(self, source_dir: str, target_dir: Path) -> None:
        if self._are_agent_logs_downloaded:
            return

        if self._environment.is_mounted:
            # For mounted environments the files are already on disk but may
            # have restrictive permissions set by the container (e.g. root:700).
            # Fix ownership and permissions inside the container (where we have
            # root privileges) so the host user can read them.
            try:
                await self._environment.exec(
                    f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(source_dir)} "
                    f"&& chmod -R u+rwX,go+rX {shlex.quote(source_dir)}"
                )
            except Exception:
                self._logger.warning(
                    f"Failed to fix permissions on {source_dir} inside container"
                )
            self._are_agent_logs_downloaded = True
            return

        try:
            await self._environment.download_dir(
                source_dir=source_dir,
                target_dir=target_dir,
            )
        except Exception:
            self._logger.error(f"Failed to download logs to {target_dir}")

        self._are_agent_logs_downloaded = True

    def _maybe_populate_agent_context(self) -> None:
        if (
            self.result.agent_result is not None
            and self.result.agent_result.is_empty()
            and isinstance(self._agent, BaseInstalledAgent)
        ):
            self._agent.populate_context_post_run(self.result.agent_result)

    async def _download_artifacts(self) -> None:
        """Download artifacts from the environment (best-effort, never fails the trial).

        Collects artifacts from two sources:
        1. Convention directory /logs/artifacts/ - auto-collected if it exists.
           Skipped for mounted environments (Docker) since it's volume-mounted.
        2. Config-driven paths from self.config.artifacts - always downloaded,
           even for mounted environments (arbitrary paths not under /logs/).
        """
        artifacts_dir = self._trial_paths.artifacts_dir
        manifest: list[dict[str, Any]] = []

        # 1. Convention directory /logs/artifacts/
        #    Skip download for mounted envs (Docker) -- already on disk via volume mount.
        #    For non-mounted envs, attempt download (silently skip if dir doesn't exist).
        if not self._environment.is_mounted:
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                await self._environment.download_dir(
                    source_dir=EnvironmentPaths.artifacts_dir.as_posix(),
                    target_dir=artifacts_dir,
                )
                manifest.append(
                    {
                        "source": EnvironmentPaths.artifacts_dir.as_posix(),
                        "destination": "artifacts",
                        "type": "directory",
                        "status": "ok",
                    }
                )
            except Exception:
                self._logger.debug(
                    "Convention artifacts dir not found or download failed (best-effort)"
                )
                manifest.append(
                    {
                        "source": EnvironmentPaths.artifacts_dir.as_posix(),
                        "destination": "artifacts",
                        "type": "directory",
                        "status": "failed",
                    }
                )

        # 2. Config-driven paths (always download, even for mounted envs,
        #    since these are arbitrary paths not under /logs/)
        for artifact in self.config.artifacts:
            # Normalize: str -> ArtifactConfig(source=str)
            if isinstance(artifact, str):
                artifact = ArtifactConfig(source=artifact)

            source = artifact.source
            dest_rel = artifact.destination or Path(source).name

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            target = artifacts_dir / dest_rel

            # Probe the environment to determine if source is a directory.
            # Fall back to suffix heuristic if the probe fails.
            is_dir: bool | None = None
            try:
                is_dir = await self._environment.is_dir(source)
            except Exception:
                is_dir = not Path(source).suffix

            try:
                if is_dir:
                    target.mkdir(parents=True, exist_ok=True)
                    await self._environment.download_dir(
                        source_dir=source, target_dir=target
                    )
                    manifest.append(
                        {
                            "source": source,
                            "destination": f"artifacts/{dest_rel}",
                            "type": "directory",
                            "status": "ok",
                        }
                    )
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    await self._environment.download_file(
                        source_path=source, target_path=target
                    )
                    manifest.append(
                        {
                            "source": source,
                            "destination": f"artifacts/{dest_rel}",
                            "type": "file",
                            "status": "ok",
                        }
                    )
            except Exception:
                self._logger.warning(
                    f"Failed to download artifact '{source}' (best-effort)"
                )
                manifest.append(
                    {
                        "source": source,
                        "destination": f"artifacts/{dest_rel}",
                        "type": "directory" if is_dir else "file",
                        "status": "failed",
                    }
                )

        # Write manifest if any artifacts were collected
        if manifest:
            try:
                self._trial_paths.artifacts_manifest_path.write_text(
                    json.dumps(manifest, indent=2)
                )
            except Exception:
                self._logger.warning("Failed to write artifacts manifest (best-effort)")

    # ---- Multi-round support ----

    async def _execute_multiround(self) -> None:
        """Execute a multi-round task: loop over rounds, each with agent work + verification."""
        start_round = self.config.verifier.multiround_start_round or 1
        max_round = self.config.verifier.multiround_max_round or self._task.num_rounds
        end_round = min(max_round, self._task.num_rounds)

        self._log_multiround_info(
            "Starting multi-round execution: rounds %s..%s (total task rounds: %s)",
            start_round,
            end_round,
            self._task.num_rounds,
        )
        if start_round > 1 and self.config.verifier.multiround_resume_source:
            resume_source_dir = Path(self.config.verifier.multiround_resume_source)
            self._log_multiround_info(
                "Resume lineage | source_trial=%s | source_dir=%s | completed_round=%s | snapshot_id=%s",
                resume_source_dir.name,
                resume_source_dir,
                start_round - 1,
                getattr(
                    self.config.verifier,
                    "multiround_resume_state_snapshot_id",
                    None,
                )
                or "unknown",
            )

        round_results: list[dict[str, Any]] = []
        timeout_exception_info: ExceptionInfo | None = None
        timeout_traceback: str | None = None

        self.result.agent_execution = TimingInfo(started_at=datetime.now(timezone.utc))
        if self.result.agent_result is None:
            self.result.agent_result = AgentContext()

        # Phase 1: fast-forward rounds 1..start_round-1
        if start_round > 1:
            resume_source = self.config.verifier.multiround_resume_source
            inplace_resume = bool(self.config.verifier.multiround_resume_trial_name)
            if not resume_source:
                self._log_multiround_info(
                    "No resume source provided; using default Oracle-prepared "
                    "late-round start for rounds 1..%s",
                    start_round - 1,
                )
            if resume_source:
                # Copy old round results from the source trial
                round_results.extend(
                    self._copy_resume_round_results(
                        Path(resume_source),
                        start_round,
                        preserve_status=inplace_resume,
                    )
                )
            # Only reconstruct historical state when this run cold-started.
            if not (resume_source and self._environment.restored_from_snapshot):
                try:
                    await self._fast_forward_rounds(start_round)
                except Exception as e:
                    self._logger.error(
                        "Fast-forward failed at rounds 1..%s: %s",
                        start_round - 1,
                        e,
                    )
                    self._aggregate_multiround_results(round_results)
                    raise
            if resume_source:
                # Restore agent session state needed for round-to-round continuation
                # (e.g., Claude Code's session files used by `--continue`).
                await self._copy_resume_agent_state(Path(resume_source), start_round)
                self._log_multiround_info(
                    "--- Resume handoff complete; live execution begins at round %s ---",
                    start_round,
                )

        # Phase 2: run agent on rounds start_round..end_round
        for round_num in range(start_round, end_round + 1):
            round_instruction = self._task.round_instruction(round_num)
            round_config = self._task.round_config(round_num)
            change_types = round_config.get("change_types", [])
            change_type = round_config.get("change_type", "unknown")
            change_types_display = ",".join(change_types) if change_types else "unknown"

            self._log_multiround_info(
                "=== Round %s/%s (change_types=%s) ===",
                round_num,
                self._task.num_rounds,
                change_types_display,
            )
            round_reward: float | int | None = None
            round_status: str | None = None
            should_stop = False

            # --- Agent work phase ---
            await self._invoke_hooks(TrialEvent.AGENT_START)

            try:
                await asyncio.wait_for(
                    self._agent.run_round(
                        instruction=round_instruction,
                        round_num=round_num,
                        environment=self._environment,
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
                    occurred_at=datetime.now(timezone.utc),
                )
                timeout_traceback = timeout_exception_info.exception_traceback
                self._logger.warning(f"Agent timed out during round {round_num}")
                round_status = "agent_timeout"
                round_results.append(
                    {
                        "round": round_num,
                        "change_type": change_type,
                        "change_types": change_types,
                        "reward": round_reward,
                        "status": round_status,
                    }
                )
                should_stop = True

            else:
                # --- Verification phase ---
                if not self.config.verifier.disable:
                    await self._invoke_hooks(TrialEvent.VERIFICATION_START)
                    try:
                        round_reward = await self._verify_round(round_num)
                        self._log_multiround_info(
                            "Round %s reward: %s", round_num, round_reward
                        )
                        round_status = "completed"
                        round_results.append(
                            {
                                "round": round_num,
                                "change_type": change_type,
                                "change_types": change_types,
                                "reward": round_reward,
                                "status": round_status,
                            }
                        )
                        if round_reward != 1:
                            self._log_multiround_info(
                                "Round %s failed (reward=%s), stopping multi-round execution.",
                                round_num,
                                round_reward,
                            )
                            should_stop = True
                    except Exception as e:
                        verification_traceback = traceback.format_exc()
                        self._log_multiround_warning(
                            "Verification did not produce a usable reward at round %s: %s. "
                            "Treating as round failure and stopping.",
                            round_num,
                            e,
                        )
                        self._trial_paths.exception_message_path.write_text(
                            verification_traceback
                        )
                        round_status = f"verification_error: {e}"
                        round_results.append(
                            {
                                "round": round_num,
                                "change_type": change_type,
                                "change_types": change_types,
                                "reward": round_reward,
                                "status": round_status,
                            }
                        )
                        should_stop = True

                    # Backup verifier outputs before next round overwrites them
                    self._backup_round_verifier_output(round_num)
                else:
                    round_status = "verification_disabled"
                    round_results.append(
                        {
                            "round": round_num,
                            "change_type": change_type,
                            "change_types": change_types,
                            "reward": round_reward,
                            "status": round_status,
                        }
                    )

            if round_status is not None:
                await self._capture_environment_state_snapshot(
                    round_num=round_num,
                    round_reward=round_reward,
                    round_status=round_status,
                )

            if should_stop:
                break

        self.result.agent_execution.finished_at = datetime.now(timezone.utc)

        # Download agent logs (once, after all rounds)
        await self._maybe_download_logs(
            source_dir=EnvironmentPaths.agent_dir.as_posix(),
            target_dir=self._trial_paths.agent_dir,
        )
        self._maybe_populate_agent_context()

        # Merge fast-forwarded rounds' trajectory steps from backup
        if start_round > 1 and self.config.verifier.multiround_resume_source:
            self._merge_resume_trajectory(
                Path(self.config.verifier.multiround_resume_source), start_round
            )

        # Aggregate per-round rewards into final VerifierResult
        self._aggregate_multiround_results(round_results)

        if timeout_exception_info is not None and self.result.exception_info is None:
            self.result.exception_info = timeout_exception_info
            if timeout_traceback is not None:
                self._trial_paths.exception_message_path.write_text(timeout_traceback)

    async def _fast_forward_rounds(self, up_to_round: int) -> None:
        """Apply oracle solve.sh for rounds 1..up_to_round-1 to bring /app/ to the correct state."""
        for ff_round in range(1, up_to_round):
            round_solution_dir = self._task.paths.round_solution_dir(ff_round)
            round_solve_path = self._task.paths.round_solve_path(ff_round)

            if not round_solve_path.exists():
                raise FileNotFoundError(
                    f"Fast-forward failed: {round_solve_path} does not exist"
                )

            await self._environment.upload_dir(
                source_dir=round_solution_dir,
                target_dir=str(EnvironmentPaths.solution_dir),
            )

            solve_script = shlex.quote(
                str(
                    EnvironmentPaths.solution_dir
                    / round_solve_path.relative_to(round_solution_dir)
                )
            )

            env = {"DEBIAN_FRONTEND": "noninteractive"}
            if self._task.config.solution.env:
                env.update(resolve_env_vars(self._task.config.solution.env))

            await self._environment.exec(
                command=f"chmod +x {solve_script} && {solve_script}",
                env=env,
            )
            self._logger.info(f"Fast-forwarded round {ff_round}")

    def _copy_resume_round_results(
        self,
        source_trial_dir: Path,
        up_to_round: int,
        *,
        preserve_status: bool = False,
    ) -> list[dict[str, Any]]:
        """Copy verifier outputs from an old trial for rounds 1..up_to_round-1.

        Returns the old round results so they can be included in the aggregation.
        """
        old_verifier = source_trial_dir / "verifier"
        new_verifier = self._trial_paths.verifier_dir

        # Copy per-round files
        for ff_round in range(1, up_to_round):
            for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
                src = old_verifier / f"round_{ff_round}_{name}"
                if src.exists():
                    shutil.copy2(src, new_verifier / f"round_{ff_round}_{name}")

        # Read old multiround_results.json and return rounds < up_to_round
        old_results_path = old_verifier / "multiround_results.json"
        if old_results_path.exists():
            all_old = json.loads(old_results_path.read_text())
            return [
                (
                    dict(rr)
                    if preserve_status
                    else {**rr, "status": "fast_forwarded"}
                )
                for rr in all_old
                if rr.get("round", 0) < up_to_round
            ]
        return []

    def _load_resume_source_agent_name(self, source_trial_dir: Path) -> str | None:
        config_path = source_trial_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            source_trial_config = TrialConfig.model_validate_json(config_path.read_text())
        except Exception as e:
            self._logger.warning(
                "Failed to read resume source config from %s: %s",
                config_path,
                e,
            )
            return None

        return source_trial_config.agent.name

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
            and self._agent.name() != AgentName.ORACLE.value
        ):
            self._log_multiround_info(
                "Resume source %s was produced by oracle; skipping %s continuation "
                "state restore and starting round %s from Oracle-prepared workspace",
                source_trial_dir,
                self._agent.name(),
                start_round,
            )
            return

        # Claude Code continuation depends on persisted session files.
        if self._agent.name() == AgentName.CLAUDE_CODE.value:
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

            target_sessions = self._trial_paths.agent_sessions_dir
            if target_sessions.exists():
                shutil.rmtree(target_sessions, ignore_errors=True)
            target_sessions.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_sessions,
                target_sessions,
                symlinks=True,
            )
            self._restore_claude_round_start_metadata(target_sessions)
            if not self._environment.is_mounted:
                await self._environment.exec(
                    f"rm -rf {shlex.quote((EnvironmentPaths.agent_dir / 'sessions').as_posix())}"
                )
                await self._environment.upload_dir(
                    source_dir=target_sessions,
                    target_dir=(EnvironmentPaths.agent_dir / "sessions").as_posix(),
                )
            self._log_multiround_info(
                "Copied Claude sessions for resume at round %s: %s -> %s",
                start_round,
                source_sessions,
                target_sessions,
            )
            return

        if self._agent.name() == AgentName.TERMINUS_2.value:
            source_state_path = self._resolve_resume_terminus_runtime_state_path(
                source_trial_dir, start_round
            )
            if source_state_path is None:
                raise RuntimeError(
                    "No matching Terminus-2 runtime snapshot was found for resume source "
                    f"{source_trial_dir} at round {start_round - 1}"
                )

            self._agent.restore_resume_state(source_state_path)
            self._log_multiround_info(
                "Restored Terminus-2 runtime state for resume at round %s: %s",
                start_round,
                source_state_path,
            )

    def _capture_pre_resume_trajectory_snapshot(
        self, source_trial_dir: Path, start_round: int
    ) -> None:
        """Persist a readable trajectory snapshot describing the pre-resume trace."""
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
            target_path = self._trial_paths.agent_pre_resume_trajectory_path(
                completed_round
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
            metadata_path = self._trial_paths.agent_pre_resume_metadata_path(
                completed_round
            )
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
        except Exception as e:
            self._logger.warning(
                "Failed to capture pre-resume trajectory snapshot from %s: %s",
                source_trajectory_path,
                e,
            )

    def _capture_pre_resume_claude_sessions_snapshot(
        self,
        source_trial_dir: Path,
        source_sessions_dir: Path,
        start_round: int,
    ) -> None:
        """Persist the exact Claude session files used for this resume."""
        completed_round = start_round - 1
        if completed_round < 1:
            return

        source_paths = TrialPaths(trial_dir=source_trial_dir)
        source_kind = (
            "round_snapshot"
            if source_sessions_dir == source_paths.agent_round_sessions_dir(completed_round)
            else "latest_sessions_fallback"
        )

        target_dir = self._trial_paths.agent_pre_resume_session_dir(completed_round)
        target_sessions_dir = self._trial_paths.agent_pre_resume_session_files_dir(
            completed_round
        )

        try:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_sessions_dir,
                target_sessions_dir,
                symlinks=True,
            )

            metadata = {
                "round": completed_round,
                "source_trial": str(source_trial_dir),
                "source_sessions_dir": str(source_sessions_dir),
                "source_kind": source_kind,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._trial_paths.agent_pre_resume_session_metadata_path(
                completed_round
            ).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        except Exception as e:
            self._logger.warning(
                "Failed to capture pre-resume Claude sessions from %s: %s",
                source_sessions_dir,
                e,
            )

    def _merge_resume_trajectory(
        self, source_trial_dir: Path, from_round: int
    ) -> None:
        """Merge fast-forwarded rounds' trajectory steps from the backup.

        Reads the backup's trajectory.json, extracts steps for rounds < from_round,
        removes inherited duplicate steps from the current trajectory, prepends the
        backup steps, and re-numbers step_ids so the merged file looks identical to
        a from-scratch run.
        """
        backup_traj_path = source_trial_dir / "agent" / "trajectory.json"
        new_traj_path = self._trial_paths.agent_dir / "trajectory.json"

        if not backup_traj_path.exists() or not new_traj_path.exists():
            self._logger.debug(
                "Skipping trajectory merge: backup or new trajectory.json missing"
            )
            return

        try:
            backup_data = json.loads(backup_traj_path.read_text())
            new_data = json.loads(new_traj_path.read_text())

            backup_steps = backup_data.get("steps", [])
            new_steps = new_data.get("steps", [])

            # Extract steps from rounds < from_round
            old_steps = [
                s for s in backup_steps
                if s.get("extra", {}).get("round") is not None
                and s["extra"]["round"] < from_round
            ]

            if not old_steps:
                self._logger.debug(
                    "No old-round steps found in backup trajectory, skipping merge"
                )
                return

            # Claude resume copies the parent session files into the child trial.
            # When the child trajectory is regenerated from those copied sessions,
            # inherited parent steps can appear again with round=null because the
            # child trial only knows its own round start timestamps. Prefer the
            # parent trajectory's original round-tagged steps and drop duplicated
            # inherited steps from the regenerated child trajectory.
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

            # Prepend old steps and re-number step_ids sequentially from 1.
            merged_steps = old_steps + filtered_new_steps
            for i, step in enumerate(merged_steps):
                step["step_id"] = i + 1

            new_data["steps"] = merged_steps
            self._annotate_trajectory_resume_context(
                trajectory_data=new_data,
                source_trial_dir=source_trial_dir,
                start_round=from_round,
            )
            new_traj_path.write_text(
                json.dumps(new_data, indent=2, ensure_ascii=False)
            )
            self._log_multiround_info(
                "Merged %s fast-forwarded steps into trajectory.json "
                "(removed %s inherited duplicates, total: %s steps)",
                len(old_steps),
                removed_inherited_steps,
                len(merged_steps),
            )
        except Exception as e:
            self._logger.warning(f"Failed to merge resume trajectory: {e}")

    @staticmethod
    def _trajectory_step_round(step: dict[str, Any]) -> int | None:
        extra = step.get("extra")
        round_num = extra.get("round") if isinstance(extra, dict) else None
        return round_num if isinstance(round_num, int) else None

    @staticmethod
    def _trajectory_step_fingerprint(
        step: dict[str, Any], *, ignore_round: bool = False
    ) -> str:
        """Return a stable fingerprint for deduplicating trajectory steps.

        ``step_id`` is intentionally ignored because the merged trajectory will be
        renumbered. ``extra.round`` is only ignored when matching inherited Claude
        session steps with ``round=null`` against the original backup step.
        """
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
        verifier_dir = self._trial_paths.verifier_dir
        for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
            (verifier_dir / name).unlink(missing_ok=True)

    def _write_verifier_exit_code(self, return_code: int) -> Path:
        exit_code_path = self._trial_paths.verifier_dir / "test-exit-code.txt"
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
        diagnostics_dir = self._trial_paths.verifier_dir
        diagnostics = f"{diagnostics_dir} ({', '.join(diagnostic_files)})"
        return f"{reason} See {diagnostics}."

    def _write_verification_error(self, message: str) -> Path:
        error_path = self._trial_paths.verifier_dir / "verification-error.txt"
        error_path.write_text(f"{message}\n")
        return error_path

    async def _verify_round(self, round_num: int) -> float | int:
        """Run verification for a specific round using round-specific tests."""
        round_tests_dir = self._task.paths.round_tests_dir(round_num)
        round_test_path = self._task.paths.round_test_path(round_num)

        # Clear previous /tests/ and upload this round's tests
        await self._environment.exec("rm -rf /tests/*")
        await self._environment.upload_dir(
            source_dir=round_tests_dir,
            target_dir="/tests",
        )

        # Clear previous reward files
        await self._environment.exec(
            "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json"
        )

        self._clear_current_round_verifier_outputs()

        # Ensure stdout/stderr paths exist on host for volume-mounted envs.
        self._trial_paths.test_stdout_path.touch()
        self._trial_paths.test_stderr_path.touch()

        # Execute test.sh
        test_script_path = shlex.quote(
            str(Path("/tests") / round_test_path.relative_to(round_tests_dir))
        )
        test_stdout_path = shlex.quote(
            str(
                EnvironmentPaths.verifier_dir
                / self._trial_paths.test_stdout_path.relative_to(
                    self._trial_paths.verifier_dir
                )
            )
        )
        test_stderr_path = shlex.quote(
            str(
                EnvironmentPaths.verifier_dir
                / self._trial_paths.test_stderr_path.relative_to(
                    self._trial_paths.verifier_dir
                )
            )
        )

        await self._environment.exec(f"chmod +x {test_script_path}")

        env = None
        if self._task.config.verifier.env:
            env = resolve_env_vars(self._task.config.verifier.env)

        try:
            exec_result = await asyncio.wait_for(
                self._environment.exec(
                    command=f"{test_script_path} > {test_stdout_path} 2> {test_stderr_path}",
                    env=env,
                ),
                timeout=self._verifier_timeout_sec,
            )
        except asyncio.TimeoutError as e:
            if not self._environment.is_mounted:
                await self._environment.download_dir(
                    source_dir=str(EnvironmentPaths.verifier_dir),
                    target_dir=self._trial_paths.verifier_dir,
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
            raise VerifierTimeoutError(message) from e

        # Download verifier dir for non-mounted envs
        if not self._environment.is_mounted:
            await self._environment.download_dir(
                source_dir=str(EnvironmentPaths.verifier_dir),
                target_dir=self._trial_paths.verifier_dir,
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

        # Read reward
        if self._trial_paths.reward_text_path.exists():
            reward_text = self._trial_paths.reward_text_path.read_text().strip()
            if not reward_text:
                message = self._build_verifier_failure_message(
                    round_num,
                    f"Reward file is empty after round {round_num}",
                    include_exit_code=True,
                )
                self._write_verification_error(message)
                raise ValueError(message)
            return float(reward_text)
        elif self._trial_paths.reward_json_path.exists():
            reward_data = json.loads(self._trial_paths.reward_json_path.read_text())
            return reward_data.get("reward", 0)
        else:
            message = self._build_verifier_failure_message(
                round_num,
                f"No reward file found after round {round_num}",
                include_exit_code=True,
            )
            self._write_verification_error(message)
            raise FileNotFoundError(message)

    def _backup_round_verifier_output(self, round_num: int) -> None:
        """Copy verifier outputs to round-specific filenames before next round overwrites."""
        verifier_dir = self._trial_paths.verifier_dir

        for name in self._MULTIROUND_VERIFIER_ARTIFACT_NAMES:
            src = verifier_dir / name
            if src.exists():
                dst = verifier_dir / f"round_{round_num}_{name}"
                shutil.copy2(src, dst)

    def _aggregate_multiround_results(
        self, round_results: list[dict[str, Any]]
    ) -> None:
        """Aggregate per-round rewards into a single VerifierResult."""
        rewards: dict[str, float | int] = {}
        round_results_by_round = {
            rr["round"]: rr
            for rr in round_results
            if isinstance(rr, dict) and isinstance(rr.get("round"), int)
        }

        start_round = self.config.verifier.multiround_start_round or 1
        configured_max_round = self.config.verifier.multiround_max_round
        task_num_rounds = getattr(getattr(self, "_task", None), "num_rounds", None)
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
            if self.config.verifier.multiround_resume_trial_name
            else start_round
        )
        aggregate_start_round = (
            getattr(self.config.verifier, "multiround_aggregate_start_round", None)
            or aggregate_start_round
        )
        aggregate_end_round = (
            getattr(self.config.verifier, "multiround_aggregate_end_round", None)
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

        if aggregate_rewards:
            rewards["reward"] = sum(aggregate_rewards) / len(aggregate_rewards)
        else:
            rewards["reward"] = 0

        self.result.verifier_result = VerifierResult(
            rewards=rewards,
            aggregate_window_start=aggregate_start_round,
            aggregate_window_end=aggregate_end_round,
        )

        # Write detailed round results
        round_results_path = self._trial_paths.verifier_dir / "multiround_results.json"
        round_results_path.write_text(json.dumps(round_results, indent=2))

    def _should_capture_state_snapshot(
        self,
        *,
        round_reward: float | int | None,
        round_status: str,
    ) -> bool:
        if not self._task.is_multiround:
            return False
        policy = self.config.verifier.multiround_state_cache_policy
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
    ) -> None:
        latest_snapshot_path = self._trial_paths.state_snapshot_path
        latest_archive_path = self._trial_paths.state_image_archive_path
        round_archive_path = self._trial_paths.round_state_image_archive_path(round_num)

        latest_payload = dict(snapshot_payload)
        latest_payload["archive_path"] = str(
            latest_archive_path.expanduser().absolute()
        )
        latest_snapshot_path.write_text(json.dumps(latest_payload, indent=2))

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

    def _requires_snapshot_capability(self) -> bool:
        if not self._task.is_multiround:
            return False
        if self.config.verifier.multiround_resume_source is not None:
            return True
        if self._resume_state_image_ref is not None:
            return True
        return self.config.verifier.multiround_state_cache_policy != "off"

    async def _capture_environment_state_snapshot(
        self,
        *,
        round_num: int,
        round_reward: float | int | None,
        round_status: str,
    ) -> None:
        if not self._should_capture_state_snapshot(
            round_reward=round_reward,
            round_status=round_status,
        ):
            return

        round_state_dir = self._trial_paths.round_state_dir(round_num)
        round_state_dir.mkdir(parents=True, exist_ok=True)

        snapshot_id = f"{self.config.trial_name}__round-{round_num}__state"
        # Terminus-2 runtime state includes live tmux/asciinema process state.
        # Docker snapshot capture stops the container, so capture this first.
        await self._capture_terminus_runtime_snapshot(round_num)
        snapshot_data = await self._environment.capture_state_snapshot(
            snapshot_id=snapshot_id,
            archive_path=self._trial_paths.round_state_image_archive_path(round_num),
            restart_container=True,
        )

        if snapshot_data is None:
            raise RuntimeError(
                "Failed to capture environment snapshot for "
                f"{self.config.trial_name} round {round_num}"
            )

        parent_snapshot_id = self._latest_snapshot_parent_id
        snapshot_payload = {
            "snapshot_id": snapshot_data.get("snapshot_id"),
            "image_ref": snapshot_data.get("image_ref"),
            "image_tag": snapshot_data.get("image_tag"),
            "archive_path": snapshot_data.get("archive_path"),
            "parent_snapshot_id": parent_snapshot_id,
            "source_trial": self.config.verifier.multiround_resume_source,
            "trial_name": self.config.trial_name,
            "round": round_num,
            "status": round_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._trial_paths.round_state_snapshot_path(round_num).write_text(
            json.dumps(snapshot_payload, indent=2)
        )
        self._write_latest_state_snapshot_alias(round_num, snapshot_payload)
        await self._capture_claude_session_snapshot(round_num)
        self._latest_snapshot_parent_id = snapshot_payload.get("snapshot_id")

        self.result.environment_state = EnvironmentStateInfo(
            snapshot_id=snapshot_payload.get("snapshot_id"),
            image_ref=snapshot_payload.get("image_tag")
            or snapshot_payload.get("image_ref"),
            image_archive=str(self._trial_paths.state_image_archive_path.absolute()),
            parent_snapshot_id=snapshot_payload.get("parent_snapshot_id"),
            source=snapshot_payload.get("source_trial"),
        )

    async def _capture_claude_session_snapshot(self, round_num: int) -> None:
        if self._agent.name() != AgentName.CLAUDE_CODE.value:
            return

        live_sessions_dir = self._trial_paths.agent_sessions_dir
        round_sessions_dir = self._trial_paths.agent_round_sessions_dir(round_num)
        remote_sessions_dir = (EnvironmentPaths.agent_dir / "sessions").as_posix()

        if not self._environment.is_mounted:
            if live_sessions_dir.exists():
                shutil.rmtree(live_sessions_dir, ignore_errors=True)
            try:
                await self._environment.download_dir(
                    source_dir=remote_sessions_dir,
                    target_dir=live_sessions_dir,
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to download Claude sessions for "
                    f"{self.config.trial_name} round {round_num}"
                ) from e
        else:
            try:
                await self._environment.exec(
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
                    )
                )
            except Exception:
                self._logger.warning(
                    "Failed to fix permissions on Claude sessions inside container: %s",
                    remote_sessions_dir,
                )

        if not live_sessions_dir.exists():
            raise RuntimeError(
                "Claude sessions directory was not found for "
                f"{self.config.trial_name} round {round_num}"
            )

        self._write_claude_round_start_metadata(live_sessions_dir)

        if round_sessions_dir.exists():
            shutil.rmtree(round_sessions_dir, ignore_errors=True)
        round_sessions_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            live_sessions_dir,
            round_sessions_dir,
            symlinks=True,
        )

    async def _capture_terminus_runtime_snapshot(self, round_num: int) -> None:
        if self._agent.name() != AgentName.TERMINUS_2.value:
            return

        snapshot_path = self._trial_paths.terminus_round_runtime_state_path(round_num)
        try:
            await self._agent.capture_resume_state(snapshot_path, round_num)
        except Exception as e:
            raise RuntimeError(
                "Failed to capture Terminus-2 runtime snapshot for "
                f"{self.config.trial_name} round {round_num}"
            ) from e

    def _write_claude_round_start_metadata(self, sessions_dir: Path) -> None:
        if self._agent.name() != AgentName.CLAUDE_CODE.value:
            return
        round_starts = getattr(self._agent, "_round_starts", None)
        if round_starts is None:
            return

        metadata_path = sessions_dir / self._CLAUDE_ROUND_STARTS_FILENAME
        payload = [
            {"round": round_num, "started_at": started_at}
            for round_num, started_at in round_starts
        ]
        metadata_path.write_text(json.dumps(payload, indent=2))

    def _restore_claude_round_start_metadata(self, sessions_dir: Path) -> None:
        if self._agent.name() != AgentName.CLAUDE_CODE.value:
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
            setattr(self._agent, "_round_starts", round_starts)

    async def run(self) -> TrialResult:
        self._trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        self._trial_paths.config_path.write_text(self.config.model_dump_json(indent=4))

        self._result = TrialResult(
            trial_name=self.config.trial_name,
            task_name=self._task.name,
            task_id=self.config.task.get_task_id(),
            started_at=datetime.now(timezone.utc),
            config=self.config,
            task_checksum=self._task.checksum,
            trial_uri=self._trial_paths.trial_dir.expanduser().resolve().as_uri(),
            agent_info=self._agent.to_agent_info(),
            source=self.config.task.source,
        )

        if self._resume_state_image_ref:
            self._result.environment_state = EnvironmentStateInfo(
                image_ref=self._resume_state_image_ref,
                image_archive=self._resume_state_archive_path,
                parent_snapshot_id=self._resume_state_snapshot_id,
                source=self.config.verifier.multiround_resume_source,
            )

        if self._requires_snapshot_capability() and not self._environment_supports_state_snapshots:
            raise RuntimeError(
                "Multi-round state snapshots are enabled but the selected environment "
                "does not implement environment state snapshots"
            )

        await self._invoke_hooks(TrialEvent.START)

        try:
            await self._setup_environment()
            await self._setup_agent()

            if self._task.is_multiround:
                # ===== Multi-round execution path =====
                await self._execute_multiround()
            else:
                # ===== Original single-round execution path =====
                try:
                    await self._execute_agent()

                    await self._maybe_download_logs(
                        source_dir=EnvironmentPaths.agent_dir.as_posix(),
                        target_dir=self._trial_paths.agent_dir,
                    )
                    self._maybe_populate_agent_context()

                except AgentTimeoutError as e:
                    self.result.exception_info = ExceptionInfo.from_exception(e)
                    self._trial_paths.exception_message_path.write_text(
                        traceback.format_exc()
                    )
                    await self._maybe_download_logs(
                        source_dir=EnvironmentPaths.agent_dir.as_posix(),
                        target_dir=self._trial_paths.agent_dir,
                    )
                    self._maybe_populate_agent_context()

                if not self.config.verifier.disable:
                    await self._run_verification()

            await self._download_artifacts()

        except asyncio.CancelledError as e:
            self._logger.debug(f"Trial {self.config.trial_name} cancelled")
            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._trial_paths.exception_message_path.write_text(
                    traceback.format_exc()
                )

            await self._download_artifacts()
            await self._invoke_hooks(TrialEvent.CANCEL)

            raise e

        except Exception as e:
            self._logger.debug(f"Trial {self.config.trial_name} failed: {e}")

            await self._maybe_download_logs(
                source_dir=EnvironmentPaths.agent_dir.as_posix(),
                target_dir=self._trial_paths.agent_dir,
            )
            self._maybe_populate_agent_context()

            if self.result.exception_info is None:
                self.result.exception_info = ExceptionInfo.from_exception(e)
                self._trial_paths.exception_message_path.write_text(
                    traceback.format_exc()
                )

            await self._download_artifacts()

        finally:
            await self._cleanup_and_finalize()
            self._close_logger_handler()

        return self.result
