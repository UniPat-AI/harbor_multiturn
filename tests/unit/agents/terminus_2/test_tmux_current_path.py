from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.terminus_2.tmux_session import TmuxSession, _ProcessInfo


def test_select_active_shell_process_prefers_deepest_descendant_shell():
    processes = {
        344: _ProcessInfo(
            pid=344,
            ppid=343,
            tty="/dev/pts/1",
            comm="bash",
            cwd="/app",
        ),
        363: _ProcessInfo(
            pid=363,
            ppid=344,
            tty="/dev/pts/1",
            comm="python3",
            cwd="/app",
        ),
        366: _ProcessInfo(
            pid=366,
            ppid=363,
            tty="/dev/pts/0",
            comm="sh",
            cwd="/app",
        ),
        367: _ProcessInfo(
            pid=367,
            ppid=366,
            tty="/dev/pts/0",
            comm="bash",
            cwd="/tmp",
        ),
    }

    selected = TmuxSession._select_active_shell_process(processes, pane_pid=344)

    assert selected is not None
    assert selected.pid == 367
    assert selected.cwd == "/tmp"


@pytest.mark.asyncio
async def test_get_current_path_uses_descendant_shell_cwd_when_recording_wraps_shell(
    temp_dir: Path,
):
    environment = AsyncMock()
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="344\n", stderr=""),
            SimpleNamespace(
                return_code=0,
                stdout=(
                    "344\t343\t/dev/pts/1\tbash\t/app\n"
                    "363\t344\t/dev/pts/1\tpython3\t/app\n"
                    "366\t363\t/dev/pts/0\tsh\t/app\n"
                    "367\t366\t/dev/pts/0\tbash\t/tmp\n"
                ),
                stderr="",
            ),
        ]
    )

    session = TmuxSession(
        session_name="terminus-2",
        environment=environment,
        logging_path=temp_dir / "tmux.log",
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
    )

    path = await session.get_current_path()

    assert path == "/tmp"
    assert environment.exec.await_count == 2


@pytest.mark.asyncio
async def test_get_current_path_falls_back_to_tmux_path_when_process_snapshot_fails(
    temp_dir: Path,
):
    environment = AsyncMock()
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="344\n", stderr=""),
            SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="/proc snapshot unavailable",
            ),
            SimpleNamespace(return_code=0, stdout="/app\n", stderr=""),
        ]
    )

    session = TmuxSession(
        session_name="terminus-2",
        environment=environment,
        logging_path=temp_dir / "tmux.log",
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
    )

    path = await session.get_current_path()

    assert path == "/app"
    assert environment.exec.await_count == 3
