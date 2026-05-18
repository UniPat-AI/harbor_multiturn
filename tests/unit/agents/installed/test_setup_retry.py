from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext


class RetryTestAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "retry-test-agent"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(environment, "curl install")

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        pass


@pytest.fixture
def retry_test_agent(temp_dir):
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir()
    return RetryTestAgent(logs_dir=logs_dir)


def _install_exec_calls(mock_environment) -> list:
    return [
        call
        for call in mock_environment.exec.call_args_list
        if call.kwargs.get("command") == "set -o pipefail; curl install"
    ]


@pytest.mark.asyncio
async def test_setup_retries_transient_install_failure(
    retry_test_agent, mock_environment, monkeypatch
):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("harbor.agents.installed.base.asyncio.sleep", sleep_mock)

    mock_environment.exec.side_effect = [
        ExecResult(return_code=0, stdout="", stderr=""),
        ExecResult(
            return_code=35,
            stdout="curl: (35) error:0A000126:SSL routines::unexpected eof while reading",
            stderr="",
        ),
        ExecResult(return_code=0, stdout="installation ok", stderr=""),
    ]

    await retry_test_agent.setup(mock_environment)

    install_calls = _install_exec_calls(mock_environment)
    assert len(install_calls) == 2
    assert sleep_mock.await_count == 1

    setup_dir = retry_test_agent.logs_dir / "setup"
    assert (setup_dir / "return-code.txt").read_text() == "0"
    assert (setup_dir / "stdout.txt").read_text() == "installation ok"
    assert not (setup_dir / "stderr.txt").exists()
    assert (setup_dir / "attempt-1" / "return-code.txt").read_text() == "35"
    assert "unexpected eof while reading" in (
        setup_dir / "attempt-1" / "stdout.txt"
    ).read_text()
    assert (setup_dir / "attempt-2" / "return-code.txt").read_text() == "0"


@pytest.mark.asyncio
async def test_setup_does_not_retry_permanent_install_failure(
    retry_test_agent, mock_environment, monkeypatch
):
    sleep_mock = AsyncMock()
    monkeypatch.setattr("harbor.agents.installed.base.asyncio.sleep", sleep_mock)

    mock_environment.exec.side_effect = [
        ExecResult(return_code=0, stdout="", stderr=""),
        ExecResult(return_code=2, stdout="", stderr="npm: command not found"),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        await retry_test_agent.setup(mock_environment)

    assert "Agent setup failed with exit code 2" in str(exc_info.value)
    assert len(_install_exec_calls(mock_environment)) == 1
    assert sleep_mock.await_count == 0
