import json
from pathlib import Path
from types import SimpleNamespace

from harbor.trial.trial import Trial
from harbor.utils.logger import logger


def _write_trajectory(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.2",
                "steps": steps,
            },
            indent=2,
        )
    )


def test_merge_resume_trajectory_deduplicates_inherited_claude_steps(tmp_path: Path):
    source_trial_dir = tmp_path / "source"
    target_trial_dir = tmp_path / "target"
    backup_traj_path = source_trial_dir / "agent" / "trajectory.json"
    new_traj_path = target_trial_dir / "agent" / "trajectory.json"

    backup_steps = [
        {
            "step_id": 1,
            "timestamp": "2026-03-06T00:00:01Z",
            "source": "user",
            "message": "round 1 prompt",
            "extra": {"round": 1},
        },
        {
            "step_id": 2,
            "timestamp": "2026-03-06T00:00:02Z",
            "source": "agent",
            "message": "round 2 response",
            "extra": {"round": 2},
        },
    ]
    _write_trajectory(backup_traj_path, backup_steps)

    # Simulate Claude resume behavior:
    # - copied parent session steps are regenerated into the child trace again,
    #   but they lose their round assignment and show up as round=null/missing
    # - the child also contains the new round-3 step
    child_steps = [
        {
            "step_id": 1,
            "timestamp": "2026-03-06T00:00:01Z",
            "source": "user",
            "message": "round 1 prompt",
        },
        {
            "step_id": 2,
            "timestamp": "2026-03-06T00:00:02Z",
            "source": "agent",
            "message": "round 2 response",
            "extra": {"round": None},
        },
        {
            "step_id": 3,
            "timestamp": "2026-03-06T00:00:03Z",
            "source": "agent",
            "message": "round 3 response",
            "extra": {"round": 3},
        },
    ]
    _write_trajectory(new_traj_path, child_steps)

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_resume_trajectory")
    trial._trial_paths = SimpleNamespace(agent_dir=target_trial_dir / "agent")

    trial._merge_resume_trajectory(source_trial_dir, from_round=3)

    merged = json.loads(new_traj_path.read_text())
    merged_steps = merged["steps"]

    assert [step["step_id"] for step in merged_steps] == [1, 2, 3]
    assert [step.get("extra", {}).get("round") for step in merged_steps] == [1, 2, 3]
    assert [step["message"] for step in merged_steps] == [
        "round 1 prompt",
        "round 2 response",
        "round 3 response",
    ]


def test_merge_resume_trajectory_preserves_legitimate_repeated_steps_in_new_round(
    tmp_path: Path,
):
    source_trial_dir = tmp_path / "source"
    target_trial_dir = tmp_path / "target"
    backup_traj_path = source_trial_dir / "agent" / "trajectory.json"
    new_traj_path = target_trial_dir / "agent" / "trajectory.json"

    backup_steps = [
        {
            "step_id": 1,
            "timestamp": "2026-03-06T00:00:01Z",
            "source": "user",
            "message": "same prompt",
            "extra": {"round": 1},
        }
    ]
    _write_trajectory(backup_traj_path, backup_steps)

    child_steps = [
        {
            "step_id": 1,
            "timestamp": "2026-03-06T00:00:01Z",
            "source": "user",
            "message": "same prompt",
        },
        {
            "step_id": 2,
            "timestamp": "2026-03-06T00:00:03Z",
            "source": "user",
            "message": "same prompt",
            "extra": {"round": 3},
        },
    ]
    _write_trajectory(new_traj_path, child_steps)

    trial = object.__new__(Trial)
    trial._logger = logger.getChild("test_trial_resume_trajectory")
    trial._trial_paths = SimpleNamespace(agent_dir=target_trial_dir / "agent")

    trial._merge_resume_trajectory(source_trial_dir, from_round=3)

    merged = json.loads(new_traj_path.read_text())
    merged_steps = merged["steps"]

    assert [step["step_id"] for step in merged_steps] == [1, 2]
    assert [step.get("extra", {}).get("round") for step in merged_steps] == [1, 3]
    assert [step["message"] for step in merged_steps] == [
        "same prompt",
        "same prompt",
    ]
