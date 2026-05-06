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


@pytest.fixture
def terminus2_instance(tmp_path: Path):
    mock_llm = MagicMock()
    mock_llm.get_model_context_limit.return_value = 128000
    mock_llm.get_model_output_limit.return_value = 4096

    with patch.object(Terminus2, "_init_llm", return_value=mock_llm):
        agent = Terminus2(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4o",
            parser_name="json",
            enable_summarize=False,
        )

    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    return agent


@pytest.mark.asyncio
async def test_capture_and_restore_resume_state_round_trip(
    terminus2_instance: Terminus2, tmp_path: Path
):
    agent = terminus2_instance
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
    agent._chat._prompt_token_ids_list = [[1, 2, 3]]
    agent._chat._completion_token_ids_list = [[4, 5]]
    agent._chat._logprobs_list = [[-0.1, -0.2]]
    agent._chat._last_response_id = "resp_123"
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
    agent._subagent_rollout_details = [{"prompt_token_ids": [[9]]}]
    agent._summarization_count = 2
    agent._api_request_times = [12.5, 18.0]
    agent._n_episodes = 4
    agent._session = SimpleNamespace(
        get_current_path=AsyncMock(return_value="/app/subdir"),
        get_previous_buffer=MagicMock(return_value="previous terminal buffer"),
    )

    snapshot_path = tmp_path / "terminus-resume.json"
    await agent.capture_resume_state(snapshot_path, round_num=2)

    with patch.object(Terminus2, "_init_llm", return_value=agent._llm):
        restored_agent = Terminus2(
            logs_dir=tmp_path / "restored-logs",
            model_name="openai/gpt-4o",
            parser_name="json",
            enable_summarize=False,
        )

    restored_agent.restore_resume_state(snapshot_path)

    assert restored_agent._chat is not None
    assert restored_agent._chat.messages == agent._chat.messages
    assert restored_agent._chat.total_input_tokens == 11
    assert restored_agent._chat.total_output_tokens == 7
    assert restored_agent._chat.total_cache_tokens == 3
    assert restored_agent._chat.total_cost == 0.42
    assert restored_agent._chat._last_response_id == "resp_123"
    assert len(restored_agent._trajectory_steps) == 1
    assert restored_agent._trajectory_steps[0].extra == {"round": 1}
    assert restored_agent._subagent_metrics.total_prompt_tokens == 5
    assert restored_agent._summarization_count == 2
    assert restored_agent._pending_resume_cwd == "/app/subdir"
    assert restored_agent._pending_resume_previous_buffer == "previous terminal buffer"
    assert restored_agent._has_pending_resume_previous_buffer is True
    assert agent._pending_resume_cwd == "/app/subdir"
    assert agent._pending_resume_previous_buffer == "previous terminal buffer"
    assert agent._has_pending_resume_previous_buffer is True


@pytest.mark.asyncio
async def test_run_round_uses_restored_state_instead_of_cold_start(
    terminus2_instance: Terminus2, tmp_path: Path
):
    snapshot_path = tmp_path / "terminus-resume.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "round": 2,
                "chat_messages": [
                    {"role": "user", "content": "round 1 prompt"},
                    {"role": "assistant", "content": "round 1 reply"},
                ],
                "chat_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "cache_tokens": 3,
                    "cost_usd": 0.42,
                },
                "chat_last_response_id": "resp_456",
                "chat_rollout_details": [{"prompt_token_ids": [[1, 2, 3]]}],
                "trajectory_steps": [
                    {
                        "step_id": 1,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "user",
                        "message": "round 1 prompt",
                        "extra": {"round": 1},
                    }
                ],
                "subagent_metrics": {
                    "total_prompt_tokens": 5,
                    "total_completion_tokens": 2,
                    "total_cached_tokens": 1,
                    "total_cost_usd": 0.05,
                },
                "subagent_rollout_details": [],
                "summarization_count": 0,
                "api_request_times_msec": [],
                "n_episodes": 0,
                "shell_current_path": "/app/subdir",
                "tmux_previous_buffer": "previous terminal buffer",
            }
        )
    )

    agent = terminus2_instance
    agent.restore_resume_state(snapshot_path)
    agent._session = SimpleNamespace(
        restore_working_directory=AsyncMock(),
        restore_previous_buffer=MagicMock(),
        get_incremental_output=AsyncMock(return_value="Current Terminal Screen:\n/app/subdir"),
    )
    agent._build_skills_section = AsyncMock(return_value=None)
    agent._run_agent_loop = AsyncMock()
    cold_start_run = AsyncMock()
    agent.run = cold_start_run

    context = AgentContext()
    await agent.run_round(
        instruction="resume from round 3",
        round_num=3,
        environment=SimpleNamespace(),
        context=context,
    )

    cold_start_run.assert_not_called()
    agent._session.restore_working_directory.assert_awaited_once_with("/app/subdir")
    agent._session.restore_previous_buffer.assert_called_once_with(
        "previous terminal buffer"
    )
    agent._run_agent_loop.assert_awaited_once()
    initial_prompt = agent._run_agent_loop.await_args.kwargs["initial_prompt"]
    assert initial_prompt == "resume from round 3\n\nCurrent Terminal Screen:\n/app/subdir"
    assert "You are an AI assistant tasked with solving command-line tasks" not in initial_prompt
    assert agent._context is context
    assert agent._trajectory_steps[-1].step_id == 2
    assert agent._trajectory_steps[-1].extra == {"round": 3}
    assert (agent.logs_dir / "trajectory.json").exists()
    assert agent._chat is not None
    assert agent._chat._last_response_id == "resp_456"


@pytest.mark.asyncio
async def test_run_round_restarts_tmux_session_after_snapshot_restart(
    terminus2_instance: Terminus2,
):
    agent = terminus2_instance
    agent._chat = Chat(agent._llm, interleaved_thinking=agent._interleaved_thinking)
    agent._pending_resume_cwd = "/app/subdir"
    agent._pending_resume_previous_buffer = "previous terminal buffer"
    agent._has_pending_resume_previous_buffer = True
    agent._session = SimpleNamespace(
        is_session_alive=AsyncMock(return_value=False),
        start=AsyncMock(),
        restore_working_directory=AsyncMock(),
        restore_previous_buffer=MagicMock(),
        get_incremental_output=AsyncMock(return_value="Current Terminal Screen:\n/app/subdir"),
    )
    agent._build_skills_section = AsyncMock(return_value=None)
    agent._run_agent_loop = AsyncMock()
    cold_start_run = AsyncMock()
    agent.run = cold_start_run

    context = AgentContext()
    await agent.run_round(
        instruction="round 2 instruction",
        round_num=2,
        environment=SimpleNamespace(),
        context=context,
    )

    cold_start_run.assert_not_called()
    agent._session.is_session_alive.assert_awaited_once()
    agent._session.start.assert_awaited_once()
    agent._session.restore_working_directory.assert_awaited_once_with("/app/subdir")
    agent._session.restore_previous_buffer.assert_called_once_with(
        "previous terminal buffer"
    )
    agent._run_agent_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_initial_run_still_uses_fresh_start_template(
    terminus2_instance: Terminus2,
):
    agent = terminus2_instance
    agent._session = SimpleNamespace(
        get_incremental_output=AsyncMock(
            return_value="Current Terminal Screen:\n/app"
        ),
    )
    agent._build_skills_section = AsyncMock(return_value=None)
    agent._run_agent_loop = AsyncMock()

    context = AgentContext()
    await agent.run(
        instruction="do the task",
        environment=SimpleNamespace(),
        context=context,
    )

    initial_prompt = agent._run_agent_loop.await_args.kwargs["initial_prompt"]
    assert initial_prompt.startswith(
        "You are an AI assistant tasked with solving command-line tasks"
    )
    assert "Task Description:\ndo the task" in initial_prompt
