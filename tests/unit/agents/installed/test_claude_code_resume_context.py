import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import TrialPaths
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.logger import logger


def _write_resume_source_config(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": "/tmp/task"},
                "trial_name": "source-trial",
                "agent": {"name": "claude-code"},
            }
        )
    )


def _make_user_event(text: str, timestamp: str) -> dict:
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": "demo-session",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def _make_assistant_event(
    text: str,
    timestamp: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": "demo-session",
        "version": "2.1.50",
        "message": {
            "model": "claude-opus-4-6",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    }


def _write_claude_sessions(
    sessions_root: Path,
    *,
    events: list[dict],
    round_starts: list[tuple[int, str]] | None = None,
) -> Path:
    project_dir = sessions_root / "projects" / "demo-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    session_file = project_dir / "session.jsonl"
    session_file.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    if round_starts is not None:
        (sessions_root / SingleStepTrial._CLAUDE_ROUND_STARTS_FILENAME).write_text(
            json.dumps(
                [
                    {"round": round_num, "started_at": started_at}
                    for round_num, started_at in round_starts
                ],
                indent=2,
            )
        )

    return session_file


def _trajectory_projection(trajectory_path: Path) -> list[dict]:
    payload = json.loads(trajectory_path.read_text())
    return [
        {
            "source": step["source"],
            "message": step["message"],
            "round": step.get("extra", {}).get("round"),
        }
        for step in payload["steps"]
    ]


@pytest.mark.asyncio
async def test_resume_context_matches_continuous_multiround_trajectory(temp_dir: Path):
    round_starts = [
        (1, "2026-03-07T10:00:00+00:00"),
        (2, "2026-03-07T10:10:00+00:00"),
        (3, "2026-03-07T10:20:00+00:00"),
    ]
    round12_events = [
        _make_user_event("round 1 prompt", "2026-03-07T10:00:00+00:00"),
        _make_assistant_event(
            "round 1 reply",
            "2026-03-07T10:00:05+00:00",
            input_tokens=10,
            output_tokens=4,
        ),
        _make_user_event("round 2 prompt", "2026-03-07T10:10:00+00:00"),
        _make_assistant_event(
            "round 2 reply",
            "2026-03-07T10:10:06+00:00",
            input_tokens=11,
            output_tokens=5,
        ),
    ]
    round3_events = [
        _make_user_event("round 3 prompt", "2026-03-07T10:20:00+00:00"),
        _make_assistant_event(
            "round 3 reply",
            "2026-03-07T10:20:07+00:00",
            input_tokens=12,
            output_tokens=6,
        ),
    ]
    all_events = round12_events + round3_events

    source_trial_dir = temp_dir / "source"
    target_trial_dir = temp_dir / "target"
    continuous_logs_dir = temp_dir / "continuous-agent"

    source_paths = TrialPaths(trial_dir=source_trial_dir)
    target_paths = TrialPaths(trial_dir=target_trial_dir)
    source_paths.mkdir()
    target_paths.mkdir()
    _write_resume_source_config(source_trial_dir)

    source_agent = ClaudeCode(
        logs_dir=source_paths.agent_dir,
        model_name="claude-opus-4-6",
    )
    source_agent._round_starts = round_starts[:2]
    _write_claude_sessions(
        source_paths.agent_sessions_dir,
        events=round12_events,
        round_starts=round_starts[:2],
    )
    _write_claude_sessions(
        source_paths.agent_round_sessions_dir(2),
        events=round12_events,
        round_starts=round_starts[:2],
    )
    source_context = AgentContext()
    source_agent.populate_context_post_run(source_context)

    continuous_agent = ClaudeCode(
        logs_dir=continuous_logs_dir,
        model_name="claude-opus-4-6",
    )
    continuous_agent._round_starts = round_starts
    _write_claude_sessions(
        continuous_logs_dir / "sessions",
        events=all_events,
        round_starts=round_starts,
    )
    continuous_context = AgentContext()
    continuous_agent.populate_context_post_run(continuous_context)

    resumed_agent = ClaudeCode(
        logs_dir=target_paths.agent_dir,
        model_name="claude-opus-4-6",
    )
    trial = object.__new__(SingleStepTrial)
    trial._logger = logger.getChild("test_claude_code_resume_context")
    trial._trial_paths = target_paths
    trial._environment = SimpleNamespace(is_mounted=True)
    trial._agent = resumed_agent
    trial.config = SimpleNamespace(verifier=SimpleNamespace())

    await trial._copy_resume_agent_state(source_trial_dir, start_round=3)

    assert resumed_agent._round_starts == round_starts[:2]
    assert (
        target_paths.agent_sessions_dir / SingleStepTrial._CLAUDE_ROUND_STARTS_FILENAME
    ).exists()
    assert target_paths.agent_pre_resume_trajectory_path(2).exists()
    assert target_paths.agent_pre_resume_session_metadata_path(2).exists()

    target_session_file = (
        target_paths.agent_sessions_dir / "projects" / "demo-project" / "session.jsonl"
    )
    with target_session_file.open("a", encoding="utf-8") as handle:
        for event in round3_events:
            handle.write(json.dumps(event) + "\n")
    resumed_agent._round_starts.append(round_starts[2])

    resumed_context = AgentContext()
    resumed_agent.populate_context_post_run(resumed_context)
    trial._merge_resume_trajectory(source_trial_dir, from_round=3)

    assert resumed_context.n_input_tokens == continuous_context.n_input_tokens
    assert resumed_context.n_output_tokens == continuous_context.n_output_tokens
    assert resumed_context.n_cache_tokens == continuous_context.n_cache_tokens
    assert _trajectory_projection(target_paths.agent_dir / "trajectory.json") == (
        _trajectory_projection(continuous_logs_dir / "trajectory.json")
    )


@pytest.mark.asyncio
async def test_run_round_writes_continue_command_for_later_rounds(temp_dir: Path):
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    agent = ClaudeCode(logs_dir=logs_dir, model_name="claude-opus-4-6")
    environment = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        )
    )
    context = AgentContext()

    await agent.run_round(
        instruction="resume into round 2",
        round_num=2,
        environment=environment,
        context=context,
    )

    command_path = logs_dir / "round-2-command-0" / "command.txt"
    assert command_path.exists()
    command = command_path.read_text()
    assert "claude --continue" in command
    assert "claude-code-round-2.txt" in command
