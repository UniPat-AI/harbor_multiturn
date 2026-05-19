"""Unit tests for DaytonaEnvironment strategy selection and DinD compose logic."""

import json
import logging
import os
import shlex
from pathlib import Path
from typing import cast

import certifi
import pytest

from harbor.environments.daytona import (
    DaytonaClientManager,
    DaytonaEnvironment,
    _DaytonaDinD,
    _DaytonaDirect,
    _DAYTONA_FORK_SOURCE_PREFIX,
    _DAYTONA_SNAPSHOT_PREFIX,
    _ensure_daytona_ssl_cert_file,
)
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


def _make_env(
    temp_dir: Path,
    *,
    compose: bool = False,
    allow_internet: bool = True,
    mounts: list[ServiceVolumeConfig] | None = None,
    extra_docker_compose: list[Path] | None = None,
):
    """Create a DaytonaEnvironment with a minimal valid setup."""
    env_dir = temp_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    if compose:
        (env_dir / "docker-compose.yaml").write_text(
            "services:\n  main:\n    build: .\n"
        )
    else:
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    if mounts is None:
        mounts = [
            {
                "type": "bind",
                "source": trial_paths.verifier_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.verifier_dir),
            },
            {
                "type": "bind",
                "source": trial_paths.agent_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.agent_dir),
            },
            {
                "type": "bind",
                "source": trial_paths.artifacts_dir.resolve().absolute().as_posix(),
                "target": str(EnvironmentPaths.artifacts_dir),
            },
        ]
    kwargs: dict = {}
    kwargs["mounts"] = mounts

    return DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="Test.Session.123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            allow_internet=allow_internet,
            cpus=2,
            memory_mb=4096,
        ),
        extra_docker_compose=extra_docker_compose,
        **kwargs,
    )


class _FakeSandbox:
    def __init__(
        self,
        sandbox_id: str = "sandbox-1",
        *,
        state: str = "stopped",
        fork_error: Exception | None = None,
        snapshot_error: Exception | None = None,
    ):
        self.id = sandbox_id
        self.name = f"name-{sandbox_id}"
        self.state = state
        self.fork_error = fork_error
        self.snapshot_error = snapshot_error
        self.started = False
        self.stopped: list[dict] = []
        self.deleted = False
        self.created_snapshots: list[tuple[str, float | None]] = []
        self.forks: list[tuple[str | None, float | None]] = []

    async def start(self, timeout=None):
        self.started = True
        self.state = "started"

    async def stop(self, timeout=None, force=False):
        self.stopped.append({"timeout": timeout, "force": force})
        self.state = "stopped"

    async def delete(self, timeout=None):
        self.deleted = True

    async def _experimental_create_snapshot(self, name, timeout=None):
        if self.snapshot_error:
            raise self.snapshot_error
        self.created_snapshots.append((name, timeout))

    async def _experimental_fork(self, name=None, timeout=None):
        self.forks.append((name, timeout))
        if self.fork_error:
            raise self.fork_error
        return _FakeSandbox("fork-1", state="started")


class _FakeDaytonaClient:
    def __init__(self, parent: _FakeSandbox):
        self.parent = parent
        self.created_params: list[object] = []

    async def get(self, sandbox_id_or_name: str):
        assert sandbox_id_or_name == self.parent.id
        return self.parent

    async def create(self, params, timeout=None):
        self.created_params.append(params)
        return _FakeSandbox("snapshot-child", state="started")


class _FakeClientManager:
    def __init__(self, client: _FakeDaytonaClient):
        self.client = client
        self.configure_calls: list[dict] = []

    async def configure(self, **kwargs):
        self.configure_calls.append(kwargs)

    async def get_client(self):
        return self.client


def test_ensure_daytona_ssl_cert_file_sets_certifi_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    _ensure_daytona_ssl_cert_file()

    assert os.environ["SSL_CERT_FILE"] == certifi.where()


def test_ensure_daytona_ssl_cert_file_preserves_user_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/ca.pem")

    _ensure_daytona_ssl_cert_file()

    assert os.environ["SSL_CERT_FILE"] == "/custom/ca.pem"


# ── Strategy selection ────────────────────────────────────────────────


class TestStrategySelection:
    def test_dockerfile_selects_direct(self, temp_dir):
        env = _make_env(temp_dir, compose=False)
        assert isinstance(env._strategy, _DaytonaDirect)
        assert not env._compose_mode

    def test_compose_selects_dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode

    def test_extra_compose_selects_dind(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(temp_dir, compose=False, extra_docker_compose=[extra])
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode

    def test_validate_raises_when_no_definition(self, temp_dir):
        env_dir = temp_dir / "empty_env"
        env_dir.mkdir()
        trial_dir = temp_dir / "trial"
        trial_dir.mkdir(exist_ok=True)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        with pytest.raises(FileNotFoundError):
            DaytonaEnvironment(
                environment_dir=env_dir,
                environment_name="bad",
                session_id="s.1",
                trial_paths=trial_paths,
                task_env_config=EnvironmentConfig(),
            )


class TestDirectMultiroundState:
    async def test_pause_fork_capture_stops_and_retains_sandbox(
        self, monkeypatch: pytest.MonkeyPatch, temp_dir
    ):
        env = _make_env(temp_dir, compose=False)
        sandbox = _FakeSandbox("parent-1", state="started")
        env._sandbox = sandbox
        env._daytona_multiround_state_mode = "pause_fork"
        archive_path = temp_dir / "state.tar.gz"

        async def fake_archive(snapshot_id, archive_path_arg):
            assert snapshot_id == "trial__round-1"
            assert archive_path_arg == archive_path
            return {"archive_path": str(archive_path)}

        monkeypatch.setattr(env, "_capture_filesystem_archive", fake_archive)

        data = await env.capture_state_snapshot(
            "trial__round-1",
            archive_path=archive_path,
        )

        assert data is not None
        assert data["provider"] == "daytona"
        assert data["provider_state_mode"] == "pause_fork"
        assert data["image_ref"] == f"{_DAYTONA_FORK_SOURCE_PREFIX}parent-1"
        assert data["archive_path"] == str(archive_path)
        assert sandbox.stopped == [{"timeout": 60, "force": False}]
        assert env._retain_sandbox_as_multiround_state is True

        await env.stop(delete=True)

        assert sandbox.deleted is False
        assert env._sandbox is None

    async def test_snapshot_capture_creates_daytona_snapshot(self, temp_dir):
        env = _make_env(temp_dir, compose=False)
        sandbox = _FakeSandbox("sandbox-1", state="started")
        env._sandbox = sandbox
        env._daytona_multiround_state_mode = "snapshot"

        data = await env.capture_state_snapshot("trial__round-1")

        assert data is not None
        assert data["provider_state_mode"] == "snapshot"
        assert data["image_ref"].startswith(_DAYTONA_SNAPSHOT_PREFIX)
        assert data["daytona_snapshot_name"]
        assert sandbox.created_snapshots == [
            (data["daytona_snapshot_name"], 300)
        ]
        assert env._retain_sandbox_as_multiround_state is False

    async def test_start_from_fork_source_forks_parent_and_resets_outputs(
        self, monkeypatch: pytest.MonkeyPatch, temp_dir
    ):
        parent = _FakeSandbox("parent-1", state="stopped")
        client = _FakeDaytonaClient(parent)
        manager = _FakeClientManager(client)

        async def fake_get_instance():
            return manager

        monkeypatch.setattr(DaytonaClientManager, "get_instance", fake_get_instance)

        env = _make_env(temp_dir, compose=False)
        env._resume_state_image_ref = f"{_DAYTONA_FORK_SOURCE_PREFIX}parent-1"
        reset_calls: list[str] = []

        async def fake_reset():
            reset_calls.append("reset")

        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDirect)
        monkeypatch.setattr(strategy, "_reset_trial_output_dirs", fake_reset)

        await env.start(force_build=False)

        assert parent.started is True
        assert parent.forks == [(env._daytona_state_name("hbfork", env.session_id), 120)]
        assert parent.stopped == []
        assert env._sandbox is not None
        assert env._sandbox.id == "fork-1"
        assert env.restored_from_snapshot is True
        assert reset_calls == ["reset"]

    async def test_start_from_fork_source_falls_back_to_snapshot_when_fork_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, temp_dir
    ):
        parent = _FakeSandbox(
            "parent-1",
            state="started",
            fork_error=RuntimeError("Cannot POST /api/sandbox/parent-1/fork"),
        )
        client = _FakeDaytonaClient(parent)
        manager = _FakeClientManager(client)

        async def fake_get_instance():
            return manager

        monkeypatch.setattr(DaytonaClientManager, "get_instance", fake_get_instance)

        env = _make_env(temp_dir, compose=False)
        env._resume_state_image_ref = f"{_DAYTONA_FORK_SOURCE_PREFIX}parent-1"
        reset_calls: list[str] = []

        async def fake_reset():
            reset_calls.append("reset")

        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDirect)
        monkeypatch.setattr(strategy, "_reset_trial_output_dirs", fake_reset)

        await env.start(force_build=False)

        assert parent.forks == [(env._daytona_state_name("hbfork", env.session_id), 120)]
        assert parent.created_snapshots == [
            (env._transient_resume_snapshot_names[0], 300)
        ]
        assert client.created_params
        assert client.created_params[0].snapshot == env._transient_resume_snapshot_names[0]
        assert env._sandbox is not None
        assert env._sandbox.id == "snapshot-child"
        assert env.restored_from_snapshot is True
        assert reset_calls == ["reset"]

    async def test_start_from_fork_source_falls_back_to_archive_when_snapshot_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, temp_dir
    ):
        parent = _FakeSandbox(
            "parent-1",
            state="started",
            fork_error=RuntimeError("Cannot POST /api/sandbox/parent-1/fork"),
            snapshot_error=RuntimeError("Cannot POST /api/sandbox/parent-1/snapshot"),
        )
        client = _FakeDaytonaClient(parent)
        manager = _FakeClientManager(client)

        async def fake_get_instance():
            return manager

        monkeypatch.setattr(DaytonaClientManager, "get_instance", fake_get_instance)

        env = _make_env(temp_dir, compose=False)
        env._resume_state_image_ref = f"{_DAYTONA_FORK_SOURCE_PREFIX}parent-1"
        env._resume_state_archive_path = str(temp_dir / "state.tar.gz")
        archive_calls: list[tuple[Path, bool]] = []

        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDirect)

        async def fake_start_from_archive(archive_path: Path, *, force_build: bool):
            archive_calls.append((archive_path, force_build))

        monkeypatch.setattr(strategy, "_start_from_archive", fake_start_from_archive)

        await env.start(force_build=False)

        assert parent.forks == [(env._daytona_state_name("hbfork", env.session_id), 120)]
        assert parent.created_snapshots == []
        assert archive_calls == [(Path(env._resume_state_archive_path), False)]

    async def test_compose_mode_rejects_multiround_state_capture(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        env._sandbox = _FakeSandbox("sandbox-1", state="started")

        with pytest.raises(RuntimeError, match="Direct Daytona"):
            await env.capture_state_snapshot("trial__round-1")


# ── DinD compose command building ─────────────────────────────────────


class TestDinDComposeCmd:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_project_name_lowercased_and_dashes(self, dind):
        assert dind._project_name == "test-session-123"

    def test_compose_cmd_is_shlex_safe(self, dind):
        cmd = dind._compose_cmd(["up", "-d"])
        # Should round-trip through shlex.split
        parts = shlex.split(cmd)
        assert parts[0] == "docker"
        assert parts[1] == "compose"
        assert "up" in parts
        assert "-d" in parts

    def test_compose_cmd_includes_project_directory(self, dind):
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        idx = parts.index("--project-directory")
        assert parts[idx + 1] == "/harbor/environment"

    def test_compose_cmd_includes_compose_files(self, dind):
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        f_indices = [i for i, p in enumerate(parts) if p == "-f"]
        file_paths = [parts[i + 1] for i in f_indices]
        assert any("docker-compose-base.yaml" in p for p in file_paths)
        assert any("docker-compose-build.yaml" in p for p in file_paths)
        assert any("docker-compose-mounts.json" in p for p in file_paths)
        assert any(
            p.endswith("/harbor/environment/docker-compose.yaml") for p in file_paths
        )

    def test_compose_cmd_uses_prebuilt_when_set(self, dind):
        dind._use_prebuilt = True
        cmd = dind._compose_cmd(["build"])
        parts = shlex.split(cmd)
        f_indices = [i for i, p in enumerate(parts) if p == "-f"]
        file_paths = [parts[i + 1] for i in f_indices]
        assert any("docker-compose-prebuilt.yaml" in p for p in file_paths)
        assert not any("docker-compose-build.yaml" in p for p in file_paths)


class TestDinDComposeFileFlags:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_flags_are_flat_list_of_pairs(self, dind):
        flags = dind._compose_file_flags()
        # Every odd index should be "-f"
        for i in range(0, len(flags), 2):
            assert flags[i] == "-f"
        # Even indices are paths
        assert len(flags) % 2 == 0

    def test_no_network_appended_when_internet_disabled(self, temp_dir):
        env = _make_env(temp_dir, compose=True, allow_internet=False)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        assert any("docker-compose-no-network.yaml" in p for p in file_paths)

    def test_no_network_absent_when_internet_allowed(self, dind):
        flags = dind._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        assert not any("docker-compose-no-network.yaml" in p for p in file_paths)

    def test_mounts_compose_positioned_between_build_and_task_compose(self, dind):
        flags = dind._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        base_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-base.yaml")
        )
        build_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-build.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        env_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("/harbor/environment/docker-compose.yaml")
        )
        assert base_idx < build_idx < mounts_idx < env_idx

    def test_extra_compose_positioned_after_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(
            temp_dir,
            compose=True,
            extra_docker_compose=[extra],
        )
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        env_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("/harbor/environment/docker-compose.yaml")
        )
        extra_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < env_idx < extra_idx

    def test_extra_compose_positioned_after_mounts_without_task_compose(self, temp_dir):
        extra = temp_dir / "extra.yaml"
        extra.write_text("services:\n  sidecar:\n    image: redis:7\n")
        env = _make_env(
            temp_dir,
            compose=False,
            extra_docker_compose=[extra],
        )
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        flags = strategy._compose_file_flags()
        file_paths = [flags[i + 1] for i in range(0, len(flags), 2)]
        extra_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-extra-0.yaml")
        )
        mounts_idx = next(
            i
            for i, p in enumerate(file_paths)
            if p.endswith("docker-compose-mounts.json")
        )
        assert mounts_idx < extra_idx


# ── DinD compose env vars ─────────────────────────────────────────────


class TestDinDComposeEnvVars:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_contains_required_keys(self, dind):
        env_vars = dind._compose_env_vars()
        required = {
            "CONTEXT_DIR",
            "MAIN_IMAGE_NAME",
            "CPUS",
            "MEMORY",
        }
        assert required <= set(env_vars.keys())

    def test_legacy_path_keys_are_self_bound(self, dind):
        env_vars = dind._compose_env_vars()
        assert env_vars["HOST_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["ENV_VERIFIER_LOGS_PATH"] == str(EnvironmentPaths.verifier_dir)
        assert env_vars["HOST_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["ENV_AGENT_LOGS_PATH"] == str(EnvironmentPaths.agent_dir)
        assert env_vars["HOST_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)
        assert env_vars["ENV_ARTIFACTS_PATH"] == str(EnvironmentPaths.artifacts_dir)

    def test_context_dir_points_to_environment(self, dind):
        assert dind._compose_env_vars()["CONTEXT_DIR"] == "/harbor/environment"

    def test_image_name_includes_env_name(self, dind):
        assert dind._compose_env_vars()["MAIN_IMAGE_NAME"] == "hb__test-task"

    def test_resources_from_config(self, dind):
        env_vars = dind._compose_env_vars()
        assert env_vars["CPUS"] == "2"
        assert env_vars["MEMORY"] == "4096M"

    def test_prebuilt_image_included_when_set(self, dind):
        dind._use_prebuilt = True
        dind._env.task_env_config = EnvironmentConfig(docker_image="myimage:latest")
        env_vars = dind._compose_env_vars()
        assert env_vars["PREBUILT_IMAGE_NAME"] == "myimage:latest"

    def test_prebuilt_image_absent_when_not_set(self, dind):
        env_vars = dind._compose_env_vars()
        assert "PREBUILT_IMAGE_NAME" not in env_vars

    def test_infra_vars_win_over_task_and_persistent_env(self, dind, caplog):
        dind._resolved_task_env = {"CPUS": "999", "CONTEXT_DIR": "/wrong"}
        dind._env._persistent_env = {"MEMORY": "1G", "MAIN_IMAGE_NAME": "wrong-image"}

        with caplog.at_level(logging.WARNING):
            env_vars = dind._compose_env_vars()

        assert env_vars["CPUS"] == "2"
        assert env_vars["MEMORY"] == "4096M"
        assert env_vars["CONTEXT_DIR"] == "/harbor/environment"
        assert env_vars["MAIN_IMAGE_NAME"] == "hb__test-task"
        assert any("CPUS" in rec.message for rec in caplog.records)


# ── DinD log path mapping ─────────────────────────────────────────────


class TestSandboxLogPath:
    @pytest.fixture
    def dind(self, temp_dir):
        env = _make_env(temp_dir, compose=True)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        return strategy

    def test_verifier_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.verifier_dir)
        assert dind._sandbox_log_path(path) == path

    def test_agent_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.agent_dir)
        assert dind._sandbox_log_path(path) == path

    def test_artifacts_dir_returns_self(self, dind):
        path = str(EnvironmentPaths.artifacts_dir)
        assert dind._sandbox_log_path(path) == path

    def test_subpath_returns_self(self, dind):
        path = str(EnvironmentPaths.verifier_dir) + "/reward.txt"
        assert dind._sandbox_log_path(path) == path

    def test_non_log_path_returns_none(self, dind):
        assert dind._sandbox_log_path("/home/user/code") is None

    def test_partial_prefix_no_match(self, dind):
        # e.g. /logs/verifier_extra should NOT match /logs/verifier
        path = str(EnvironmentPaths.verifier_dir) + "_extra"
        assert dind._sandbox_log_path(path) is None


# ── Self-bind volume resolution ───────────────────────────────────────


class TestResolveVolumes:
    def test_self_binds_trial_bind_mounts(self, temp_dir):
        mounts: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/host/never/applies/agent",
                "target": str(EnvironmentPaths.agent_dir),
            },
            {
                "type": "bind",
                "source": "/host/never/applies/verifier",
                "target": str(EnvironmentPaths.verifier_dir),
            },
        ]
        env = _make_env(temp_dir, compose=True, mounts=mounts)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        volumes = strategy._resolve_volumes()
        assert [v["source"] for v in volumes] == [v["target"] for v in volumes]
        assert {v["target"] for v in volumes} == {
            str(EnvironmentPaths.agent_dir),
            str(EnvironmentPaths.verifier_dir),
        }

    def test_self_binds_every_mount(self, temp_dir):
        """Every bind mount in `mounts` (base or user-additive) gets
        self-bound — the trial now passes the combined list."""
        combined: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/discarded",
                "target": str(EnvironmentPaths.verifier_dir),
            },
            {
                "type": "bind",
                "source": "/discarded",
                "target": "/in/container/extra",
            },
        ]
        env = _make_env(temp_dir, compose=True, mounts=combined)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)
        volumes = strategy._resolve_volumes()
        assert [v["source"] for v in volumes] == [v["target"] for v in volumes]


class TestStageMountsComposeFile:
    async def test_writes_json_locally_and_uploads_to_vm(self, temp_dir):
        mounts: list[ServiceVolumeConfig] = [
            {
                "type": "bind",
                "source": "/discarded",
                "target": str(EnvironmentPaths.verifier_dir),
            }
        ]
        env = _make_env(temp_dir, compose=True, mounts=mounts)
        strategy = env._strategy
        assert isinstance(strategy, _DaytonaDinD)

        uploaded: list[tuple[str, str, dict]] = []

        async def _fake_upload(source, target):
            source = Path(source)
            assert source.name == "docker-compose-mounts.json"
            assert source.parent != env.trial_paths.trial_dir
            uploaded.append((str(source), target, json.loads(source.read_text())))

        env._sdk_upload_file = _fake_upload  # type: ignore[method-assign]

        volumes = strategy._resolve_volumes()
        await strategy._stage_mounts_compose_file(volumes)

        source, target, body = uploaded[0]
        assert not Path(source).exists()
        assert not list(env.trial_paths.trial_dir.glob("*docker-compose-mounts.json"))
        assert body["services"]["main"]["volumes"] == cast(list, volumes)

        # Uploaded under the shared compose dir on the VM with the canonical name.
        assert target == "/harbor/compose/docker-compose-mounts.json"


# ── _sandbox_exec shell parameter ─────────────────────────────────────


class TestSandboxExecShellParam:
    def test_direct_strategy_properties(self, temp_dir):
        """Direct strategy should use default shell (bash -lc)."""
        env = _make_env(temp_dir, compose=False)
        assert isinstance(env._strategy, _DaytonaDirect)

    def test_dind_strategy_properties(self, temp_dir):
        """DinD strategy should exist and have compose mode."""
        env = _make_env(temp_dir, compose=True)
        assert isinstance(env._strategy, _DaytonaDinD)
        assert env._compose_mode


# ── Client configuration kwarg plumbing ───────────────────────────────


class _StubClientManager:
    """Records calls to ``configure`` without spinning up a real client."""

    def __init__(self):
        self.configure_calls: list[dict] = []

    async def configure(self, **kwargs):
        self.configure_calls.append(kwargs)


class TestConfigureDaytonaClient:
    async def test_absent_kwarg_does_not_call_configure(self, temp_dir):
        env = _make_env(temp_dir)
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == []

    async def test_int_kwarg_forwards_to_configure(self, temp_dir):
        env = _make_env(temp_dir)
        env._kwargs["connection_pool_maxsize"] = 500
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == [{"connection_pool_maxsize": 500}]

    async def test_none_kwarg_forwards_explicit_none(self, temp_dir):
        env = _make_env(temp_dir)
        env._kwargs["connection_pool_maxsize"] = None
        stub = _StubClientManager()
        env._client_manager = stub
        await env._configure_daytona_client()
        assert stub.configure_calls == [{"connection_pool_maxsize": None}]


# ── DaytonaClientManager first-wins semantics ─────────────────────────


class TestDaytonaClientManagerConfigure:
    async def test_first_call_stores_value(self):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        assert mgr._client_config_set is True
        assert mgr._connection_pool_maxsize == 500

    async def test_repeated_same_value_is_silent(self, caplog):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=500)
        assert caplog.records == []
        assert mgr._connection_pool_maxsize == 500

    async def test_conflicting_value_warns_and_keeps_first(self, caplog):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=500)
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=999)
        assert "already configured" in caplog.text
        assert mgr._connection_pool_maxsize == 500

    async def test_configure_after_client_built_warns(self, caplog):
        mgr = DaytonaClientManager()
        # Simulate a client that was built before any configure() call.
        # configure() only checks ``is not None``; it never dereferences.
        mgr._client = object()  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING):
            await mgr.configure(connection_pool_maxsize=500)
        assert "before any explicit configuration" in caplog.text
        assert mgr._client_config_set is False
        assert mgr._connection_pool_maxsize is None

    async def test_explicit_none_is_preserved(self):
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=None)
        assert mgr._client_config_set is True
        assert mgr._connection_pool_maxsize is None

    async def test_cleanup_resets_config_so_reconfigure_takes_effect(self):
        """Cleanup must clear recorded config; otherwise a process that closes
        and reopens the client (notebooks, test suites, library embedding)
        would keep using the first-ever value even after reconfiguration."""
        mgr = DaytonaClientManager()
        await mgr.configure(connection_pool_maxsize=5)
        await mgr._cleanup()
        assert mgr._client_config_set is False
        assert mgr._connection_pool_maxsize is None
        await mgr.configure(connection_pool_maxsize=9)
        assert mgr._connection_pool_maxsize == 9
