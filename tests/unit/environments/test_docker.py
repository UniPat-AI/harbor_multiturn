"""Unit tests for DockerEnvironment command construction."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


@pytest.fixture
def docker_env(temp_dir):
    """Create a DockerEnvironment with a minimal valid setup."""
    env_dir = temp_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    return DockerEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="test-task__abc123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(docker_image="ubuntu:22.04"),
    )


class TestUploadDir:
    """Tests for the /. suffix fix in upload_dir."""

    async def test_upload_dir_appends_dot_suffix(self, docker_env):
        """upload_dir should append /. to source_dir so docker cp copies contents,
        not the directory itself, avoiding nested directories when target exists."""
        docker_env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )

        await docker_env.upload_dir("/local/tests", "/tests")

        docker_env._run_docker_compose_command.assert_called_once_with(
            ["cp", "/local/tests/.", "main:/tests"],
            check=True,
        )

    async def test_upload_dir_with_path_object(self, docker_env):
        """upload_dir should handle Path objects correctly."""
        docker_env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )

        await docker_env.upload_dir(Path("/local/solution"), "/solution")

        docker_env._run_docker_compose_command.assert_called_once_with(
            ["cp", "/local/solution/.", "main:/solution"],
            check=True,
        )


class TestDownloadDir:
    """Tests for the /. suffix fix in download_dir."""

    async def test_download_dir_appends_dot_suffix(self, docker_env):
        """download_dir should append /. to the container source path."""
        docker_env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )
        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))

        await docker_env.download_dir("/tests", "/local/tests")

        docker_env._run_docker_compose_command.assert_called_once_with(
            ["cp", "main:/tests/.", "/local/tests"],
            check=True,
        )

    async def test_download_dir_with_path_target(self, docker_env):
        """download_dir should handle Path objects for target_dir."""
        docker_env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )
        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))

        await docker_env.download_dir("/logs/agent", Path("/local/agent"))

        docker_env._run_docker_compose_command.assert_called_once_with(
            ["cp", "main:/logs/agent/.", "/local/agent"],
            check=True,
        )


class TestChownBeforeDownload:
    """Tests for best-effort chown before docker compose cp."""

    @patch("harbor.environments.docker.docker.os.getuid", return_value=1000)
    @patch("harbor.environments.docker.docker.os.getgid", return_value=1000)
    async def test_download_file_runs_chown_before_cp(
        self, _getgid, _getuid, docker_env
    ):
        """download_file should exec chown before running docker compose cp."""
        calls: list[str] = []

        async def track_exec(command, **kwargs):
            calls.append(f"exec:{command}")
            return ExecResult(return_code=0)

        async def track_cp(command, **kwargs):
            calls.append(f"compose:{command}")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(side_effect=track_exec)
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_cp)

        await docker_env.download_file("/app/result.txt", "/local/result.txt")

        assert len(calls) == 2
        assert calls[0] == "exec:chown 1000:1000 /app/result.txt"
        assert calls[1].startswith("compose:")

    @patch("harbor.environments.docker.docker.os.getuid", return_value=501)
    @patch("harbor.environments.docker.docker.os.getgid", return_value=20)
    async def test_download_dir_runs_recursive_chown_before_cp(
        self, _getgid, _getuid, docker_env
    ):
        """download_dir should exec chown -R before running docker compose cp."""
        calls: list[str] = []

        async def track_exec(command, **kwargs):
            calls.append(f"exec:{command}")
            return ExecResult(return_code=0)

        async def track_cp(command, **kwargs):
            calls.append(f"compose:{command}")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(side_effect=track_exec)
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_cp)

        await docker_env.download_dir("/logs", "/local/logs")

        assert len(calls) == 3
        assert calls[0] == "exec:chown -R 501:20 /logs"
        assert calls[1] == "exec:chmod -R u+rwX,go+rX /logs"
        assert calls[2].startswith("compose:")

    @patch("harbor.environments.docker.docker.os.getuid", return_value=1000)
    @patch("harbor.environments.docker.docker.os.getgid", return_value=1000)
    async def test_download_proceeds_when_chown_fails(
        self, _getgid, _getuid, docker_env
    ):
        """Download should still succeed even if chown exec fails."""
        docker_env.exec = AsyncMock(
            return_value=ExecResult(return_code=1, stdout="Operation not permitted")
        )
        docker_env._run_docker_compose_command = AsyncMock(
            return_value=ExecResult(return_code=0)
        )

        await docker_env.download_file("/app/file.txt", "/local/file.txt")

        docker_env._run_docker_compose_command.assert_called_once()

    async def test_chown_is_noop_without_getuid(self, docker_env):
        """_chown_to_host_user should be a no-op when os.getuid is unavailable."""
        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))

        # Simulate Windows by making hasattr(os, "getuid") return False
        with patch("harbor.environments.docker.docker.os") as mock_os:
            del mock_os.getuid
            await docker_env._chown_to_host_user("/some/path")

        docker_env.exec.assert_not_called()


class TestStartStaleContainerCleanup:
    """Tests for the stale container cleanup in start()."""

    async def test_start_runs_down_before_up(self, docker_env):
        """start() should run 'down --remove-orphans' before 'up -d'."""
        calls = []

        async def track_calls(command, **kwargs):
            calls.append(command)
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        await docker_env.start(force_build=False)

        assert calls == [
            ["down", "--remove-orphans"],
            ["up", "--detach", "--wait"],
        ]

    async def test_start_with_build_runs_down_before_up(self, docker_env):
        """start(force_build=True) should build, then down, then up."""
        calls = []

        async def track_calls(command, **kwargs):
            calls.append(command)
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        await docker_env.start(force_build=True)

        assert calls == [
            ["build"],
            ["down", "--remove-orphans"],
            ["up", "--detach", "--wait"],
        ]

    async def test_start_proceeds_when_down_fails(self, docker_env):
        """start() should still attempt 'up -d' even if 'down' fails."""
        calls = []

        async def track_calls(command, **kwargs):
            calls.append(command)
            if command == ["down", "--remove-orphans"]:
                raise RuntimeError("No such container")
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        await docker_env.start(force_build=False)

        assert calls == [
            ["down", "--remove-orphans"],
            ["up", "--detach", "--wait"],
        ]

    async def test_start_propagates_up_failure(self, docker_env):
        """start() should propagate errors from 'up -d'."""

        async def track_calls(command, **kwargs):
            if command == ["up", "--detach", "--wait"]:
                raise RuntimeError("Container creation failed")
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        with pytest.raises(RuntimeError, match="Container creation failed"):
            await docker_env.start(force_build=False)


class TestStartFromResumeSnapshot:
    """Tests for DockerEnvironment resume snapshot restore semantics."""

    async def test_start_uses_resume_snapshot_without_build_even_if_force_build(
        self, docker_env
    ):
        """If a resume snapshot image is available, start() should restore from it
        and skip docker compose build, even when force_build=True."""
        calls = []
        docker_env._resume_state_image_ref = "hbstate__round-2"
        docker_env._resume_state_archive_path = None
        docker_env._docker_image_exists = AsyncMock(return_value=True)

        async def track_calls(command, **kwargs):
            calls.append(command)
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        await docker_env.start(force_build=True)

        assert docker_env.restored_from_snapshot is True
        assert docker_env._use_prebuilt is True
        assert docker_env._env_vars.prebuilt_image_name == "hbstate__round-2"
        assert calls == [
            ["down", "--remove-orphans"],
            ["up", "--detach", "--wait"],
        ]

    async def test_start_loads_snapshot_archive_before_up_when_image_missing(
        self, docker_env, temp_dir
    ):
        """If the snapshot image is missing locally but the archive exists, start()
        should load the archive and still skip build."""
        calls = []
        archive_path = temp_dir / "round-2.tar"
        archive_path.write_text("snapshot-archive")

        image_checks = [False, True]

        async def track_image_exists(image_ref):
            return image_checks.pop(0)

        async def track_calls(command, **kwargs):
            calls.append(command)
            return ExecResult(return_code=0)

        docker_env._resume_state_image_ref = "hbstate__round-2"
        docker_env._resume_state_archive_path = archive_path
        docker_env._docker_image_exists = AsyncMock(side_effect=track_image_exists)
        docker_env._load_image_archive = AsyncMock()
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_calls)

        await docker_env.start(force_build=False)

        docker_env._load_image_archive.assert_called_once_with(archive_path)
        assert docker_env.restored_from_snapshot is True
        assert docker_env._use_prebuilt is True
        assert calls == [
            ["down", "--remove-orphans"],
            ["up", "--detach", "--wait"],
        ]


class TestStopChownBindMounts:
    """Tests for best-effort chown of bind-mounted /logs before stop."""

    @patch("harbor.environments.docker.docker.os.getuid", return_value=1000)
    @patch("harbor.environments.docker.docker.os.getgid", return_value=1000)
    async def test_stop_runs_chown_before_down(self, _getgid, _getuid, docker_env):
        """stop() should exec chown -R on /logs before docker compose down."""
        calls: list[str] = []

        async def track_exec(command, **kwargs):
            calls.append(f"exec:{command}")
            return ExecResult(return_code=0)

        async def track_compose(command, **kwargs):
            calls.append(f"compose:{command}")
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(side_effect=track_exec)
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)

        await docker_env.stop(delete=False)

        assert calls[0] == "compose:['ps', '-q', 'main']"
        assert calls[1] == "exec:chown -R 1000:1000 /logs"
        assert calls[2] == "exec:chmod -R u+rwX,go+rX /logs"
        assert any("compose:['down']" in c for c in calls[3:])

    @patch("harbor.environments.docker.docker.os.getuid", return_value=1000)
    @patch("harbor.environments.docker.docker.os.getgid", return_value=1000)
    async def test_stop_proceeds_when_chown_fails(self, _getgid, _getuid, docker_env):
        """stop() should still run docker compose down even if chown exec fails."""
        async def track_compose(command, **kwargs):
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(
            return_value=ExecResult(return_code=1, stdout="Operation not permitted")
        )
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)

        await docker_env.stop(delete=False)

        docker_env._run_docker_compose_command.assert_any_call(["down"])

    async def test_stop_skips_permission_fix_when_container_not_running(self, docker_env):
        """stop() should skip exec-based permission fixes if main is already stopped."""
        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))

        async def track_compose(command, **kwargs):
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="")
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)

        await docker_env.stop(delete=False)

        docker_env.exec.assert_not_called()
        docker_env._run_docker_compose_command.assert_any_call(["down"])

    async def test_stop_with_delete_removes_images_for_compatibility(self, docker_env):
        """delete=True should preserve the legacy image cleanup behavior."""
        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))

        async def track_compose(command, **kwargs):
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="")
            return ExecResult(return_code=0)

        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)

        await docker_env.stop(delete=True)

        docker_env.exec.assert_not_called()
        docker_env._run_docker_compose_command.assert_any_call(
            ["down", "--rmi", "all", "--volumes", "--remove-orphans"]
        )


class TestCaptureStateSnapshot:
    """Tests for consistent snapshot capture from Docker environments."""

    async def test_capture_snapshot_stops_running_container_before_commit(self, docker_env):
        """capture_state_snapshot should sync, stop, then commit the stopped container."""
        compose_calls = []
        docker_calls = []

        async def track_compose(command, **kwargs):
            compose_calls.append(command)
            if command == ["ps", "-a", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            return ExecResult(return_code=0)

        async def track_docker(command, **kwargs):
            docker_calls.append(command)
            if command[:3] == ["image", "inspect", "--format"]:
                return ExecResult(return_code=0, stdout="sha256:imageid\n")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)
        docker_env._run_docker_command = AsyncMock(side_effect=track_docker)

        archive_path = docker_env.trial_paths.state_dir / "snapshot-image.tar"
        result = await docker_env.capture_state_snapshot(
            snapshot_id="trial__state",
            archive_path=archive_path,
        )

        assert result is not None
        assert compose_calls == [
            ["ps", "-a", "-q", "main"],
            ["ps", "-q", "main"],
            ["stop", "-t", "30", "main"],
            ["ps", "-a", "-q", "main"],
        ]
        docker_env.exec.assert_called_once_with("sync")
        assert docker_calls[0] == ["commit", "container123", "hbstate__trial--state"]
        assert docker_calls[1][:3] == ["image", "inspect", "--format"]
        assert docker_calls[2][:2] == ["save", "-o"]

    async def test_capture_snapshot_commits_stopped_container_without_extra_stop(
        self, docker_env
    ):
        """capture_state_snapshot should commit an already-stopped container directly."""
        compose_calls = []
        docker_calls = []

        async def track_compose(command, **kwargs):
            compose_calls.append(command)
            if command == ["ps", "-a", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="")
            return ExecResult(return_code=0)

        async def track_docker(command, **kwargs):
            docker_calls.append(command)
            if command[:3] == ["image", "inspect", "--format"]:
                return ExecResult(return_code=0, stdout="sha256:imageid\n")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)
        docker_env._run_docker_command = AsyncMock(side_effect=track_docker)

        result = await docker_env.capture_state_snapshot(snapshot_id="trial__state")

        assert result is not None
        assert compose_calls == [
            ["ps", "-a", "-q", "main"],
            ["ps", "-q", "main"],
        ]
        docker_env.exec.assert_not_called()
        assert docker_calls[0] == ["commit", "container123", "hbstate__trial--state"]

    async def test_capture_snapshot_returns_none_when_container_missing(self, docker_env):
        """capture_state_snapshot should return None when no main container exists."""

        async def track_compose(command, **kwargs):
            assert command == ["ps", "-a", "-q", "main"]
            return ExecResult(return_code=0, stdout="")

        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)
        docker_env._run_docker_command = AsyncMock(return_value=ExecResult(return_code=0))

        result = await docker_env.capture_state_snapshot(snapshot_id="trial__state")

        assert result is None
        docker_env.exec.assert_not_called()
        docker_env._run_docker_command.assert_not_called()

    async def test_capture_snapshot_restarts_container_when_requested(self, docker_env):
        """capture_state_snapshot should bring the environment back when requested."""
        compose_calls = []
        docker_calls = []

        async def track_compose(command, **kwargs):
            compose_calls.append(command)
            if command == ["ps", "-a", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            if command == ["ps", "-q", "main"]:
                return ExecResult(return_code=0, stdout="container123\n")
            return ExecResult(return_code=0)

        async def track_docker(command, **kwargs):
            docker_calls.append(command)
            if command[:3] == ["image", "inspect", "--format"]:
                return ExecResult(return_code=0, stdout="sha256:imageid\n")
            return ExecResult(return_code=0)

        docker_env.exec = AsyncMock(return_value=ExecResult(return_code=0))
        docker_env._run_docker_compose_command = AsyncMock(side_effect=track_compose)
        docker_env._run_docker_command = AsyncMock(side_effect=track_docker)

        archive_path = docker_env.trial_paths.state_dir / "snapshot-image.tar"
        result = await docker_env.capture_state_snapshot(
            snapshot_id="trial__state",
            archive_path=archive_path,
            restart_container=True,
        )

        assert result is not None
        assert compose_calls == [
            ["ps", "-a", "-q", "main"],
            ["ps", "-q", "main"],
            ["stop", "-t", "30", "main"],
            ["ps", "-a", "-q", "main"],
            ["up", "--detach", "--wait"],
        ]
        docker_env.exec.assert_called_once_with("sync")
        assert docker_calls[0] == ["commit", "container123", "hbstate__trial--state"]
