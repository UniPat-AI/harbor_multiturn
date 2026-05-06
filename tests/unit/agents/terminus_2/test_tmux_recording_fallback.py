from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.terminus_2.tmux_session import TmuxSession


@pytest.mark.asyncio
async def test_start_disables_recording_when_asciinema_start_keeps_failing(
    temp_dir: Path,
):
    environment = AsyncMock()
    environment.exec = AsyncMock(
        side_effect=[
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
        ]
    )
    environment.upload_file = AsyncMock()
    environment.session_id = "test-session"

    session = TmuxSession(
        session_name="terminus-2",
        environment=environment,
        logging_path=temp_dir / "tmux.log",
        local_asciinema_recording_path=temp_dir / "recording.cast",
        remote_asciinema_recording_path=Path("/logs/agent/recording.cast"),
    )
    session._attempt_tmux_installation = AsyncMock()
    session.send_keys = AsyncMock(side_effect=RuntimeError("tmux send-keys failed"))

    await session.start()

    assert session._disable_recording is True
    assert session._remote_asciinema_recording_path is None
    assert session._local_asciinema_recording_path is None
    assert session.send_keys.await_count == 3
    environment.upload_file.assert_not_awaited()
