import logging
from unittest.mock import AsyncMock, Mock

import pytest

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2


@pytest.mark.asyncio
async def test_send_keys_recovers_dead_tmux_session_once():
    agent = Terminus2.__new__(Terminus2)
    agent.logger = logging.getLogger("test")
    agent._pending_resume_cwd = None
    agent._pending_resume_previous_buffer = None
    agent._has_pending_resume_previous_buffer = False

    session = Mock()
    session.get_current_path = AsyncMock(return_value="/app/work")
    session.get_previous_buffer = Mock(return_value="previous terminal output")
    session.send_keys = AsyncMock(side_effect=[RuntimeError("tmux send failed"), None])
    session.is_session_alive = AsyncMock(return_value=False)
    session.start = AsyncMock()
    session.restore_working_directory = AsyncMock()
    session.restore_previous_buffer = Mock()

    agent._session = session

    await agent._send_keys_with_session_recovery(
        session,
        Command(keystrokes="echo ok", duration_sec=0.0),
    )

    assert session.send_keys.await_count == 2
    session.start.assert_awaited_once()
    session.restore_working_directory.assert_awaited_once_with("/app/work")
    session.restore_previous_buffer.assert_called_once_with("previous terminal output")


@pytest.mark.asyncio
async def test_send_keys_does_not_restart_live_session_on_success():
    agent = Terminus2.__new__(Terminus2)
    agent.logger = logging.getLogger("test")
    agent._pending_resume_cwd = None
    agent._pending_resume_previous_buffer = None
    agent._has_pending_resume_previous_buffer = False

    session = Mock()
    session.get_current_path = AsyncMock(return_value="/app/work")
    session.get_previous_buffer = Mock(return_value="previous terminal output")
    session.send_keys = AsyncMock(return_value=None)
    session.is_session_alive = AsyncMock(return_value=True)
    session.start = AsyncMock()
    session.restore_working_directory = AsyncMock()
    session.restore_previous_buffer = Mock()

    agent._session = session

    await agent._send_keys_with_session_recovery(
        session,
        Command(keystrokes="echo ok", duration_sec=0.0),
    )

    session.send_keys.assert_awaited_once()
    session.start.assert_not_awaited()
    session.restore_working_directory.assert_not_awaited()
    session.restore_previous_buffer.assert_not_called()
