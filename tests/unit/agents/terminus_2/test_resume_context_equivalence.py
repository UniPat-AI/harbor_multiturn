import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step


def _make_agent(logs_dir: Path) -> Terminus2:
    mock_llm = MagicMock()
    mock_llm.get_model_context_limit.return_value = 128000
    mock_llm.get_model_output_limit.return_value = 4096

    with patch.object(Terminus2, "_init_llm", return_value=mock_llm):
        agent = Terminus2(
            logs_dir=logs_dir,
            model_name="openai/gpt-4o",
            parser_name="json",
            enable_summarize=False,
        )

    logs_dir.mkdir(parents=True, exist_ok=True)
    return agent


def _seed_round_one_state(agent: Terminus2) -> None:
    agent._chat = Chat(agent._llm, interleaved_thinking=agent._interleaved_thinking)
    agent._chat.messages.extend(
        [
            {"role": "user", "content": "round 1 prompt"},
            {"role": "assistant", "content": "round 1 reply"},
        ]
    )
    agent._chat._cumulative_input_tokens = 11
    agent._chat._cumulative_output_tokens = 7
    agent._chat._cumulative_cache_tokens = 3
    agent._chat._cumulative_cost = 0.42
    agent._chat._last_response_id = "resp_456"
    agent._chat._prompt_token_ids_list = [[1, 2, 3]]
    agent._trajectory_steps = [
        Step(
            step_id=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="user",
            message="round 1 prompt",
            extra={"round": 1},
        )
    ]
    agent._subagent_metrics.total_prompt_tokens = 5
    agent._subagent_metrics.total_completion_tokens = 2
    agent._subagent_metrics.total_cached_tokens = 1
    agent._subagent_metrics.total_cost_usd = 0.05
    agent._api_request_times = [12.5, 18.0]
    agent._n_episodes = 4


@pytest.mark.asyncio
async def test_restored_round_prompt_matches_continuous_execution(temp_dir: Path):
    snapshot_source = _make_agent(temp_dir / "snapshot-source")
    _seed_round_one_state(snapshot_source)
    snapshot_source._session = SimpleNamespace(
        get_current_path=AsyncMock(return_value="/app/subdir"),
        get_previous_buffer=MagicMock(return_value="previous terminal buffer"),
    )
    snapshot_path = temp_dir / "terminus-resume.json"
    await snapshot_source.capture_resume_state(snapshot_path, round_num=1)

    continuous_agent = _make_agent(temp_dir / "continuous")
    _seed_round_one_state(continuous_agent)
    continuous_session = SimpleNamespace(
        get_incremental_output=AsyncMock(
            return_value="Current Terminal Screen:\n/app/subdir"
        ),
        restore_working_directory=AsyncMock(),
        restore_previous_buffer=MagicMock(),
    )
    continuous_agent._session = continuous_session
    continuous_agent._build_skills_section = AsyncMock(return_value=None)
    continuous_agent._run_agent_loop = AsyncMock()

    restored_agent = _make_agent(temp_dir / "restored")
    restored_agent.restore_resume_state(snapshot_path)
    restored_session = SimpleNamespace(
        get_incremental_output=AsyncMock(
            return_value="Current Terminal Screen:\n/app/subdir"
        ),
        restore_working_directory=AsyncMock(),
        restore_previous_buffer=MagicMock(),
    )
    restored_agent._session = restored_session
    restored_agent._build_skills_section = AsyncMock(return_value=None)
    restored_agent._run_agent_loop = AsyncMock()

    continuous_context = AgentContext()
    restored_context = AgentContext()

    await continuous_agent.run_round(
        instruction="resume from round 2",
        round_num=2,
        environment=SimpleNamespace(),
        context=continuous_context,
    )
    await restored_agent.run_round(
        instruction="resume from round 2",
        round_num=2,
        environment=SimpleNamespace(),
        context=restored_context,
    )

    continuous_prompt = continuous_agent._run_agent_loop.await_args.kwargs[
        "initial_prompt"
    ]
    restored_prompt = restored_agent._run_agent_loop.await_args.kwargs["initial_prompt"]
    continuous_chat = continuous_agent._run_agent_loop.await_args.kwargs["chat"]
    restored_chat = restored_agent._run_agent_loop.await_args.kwargs["chat"]

    expected_prompt = "resume from round 2\n\nCurrent Terminal Screen:\n/app/subdir"
    assert continuous_prompt == restored_prompt
    assert continuous_prompt == expected_prompt
    assert "You are an AI assistant tasked with solving command-line tasks" not in continuous_prompt
    assert continuous_chat.messages == restored_chat.messages
    assert (
        continuous_agent._trajectory_steps[-1].message
        == restored_agent._trajectory_steps[-1].message
    )
    assert continuous_agent._trajectory_steps[-1].extra == {"round": 2}
    assert restored_agent._trajectory_steps[-1].extra == {"round": 2}

    assert continuous_context.model_dump() == restored_context.model_dump()
    continuous_last_message = json.loads(
        (continuous_agent.logs_dir / "trajectory.json").read_text()
    )["steps"][-1]["message"]
    restored_last_message = json.loads(
        (restored_agent.logs_dir / "trajectory.json").read_text()
    )["steps"][-1]["message"]
    assert continuous_last_message == restored_last_message

    continuous_session.restore_working_directory.assert_not_called()
    continuous_session.restore_previous_buffer.assert_not_called()
    restored_session.restore_working_directory.assert_awaited_once_with("/app/subdir")
    restored_session.restore_previous_buffer.assert_called_once_with(
        "previous terminal buffer"
    )
