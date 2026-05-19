# Multiturn Runtime Design

## Goal

The overlay turns a Harbor task into a continuous multi-round Terminal-Bench evaluation. A single trial advances the same workspace through a sequence of evolving requirements, verifies each round at the round boundary, and records enough state to resume or fan out later attempts from a stable boundary.

## Task Semantics

The task directory contains one shared environment and a sequence of round directories:

```text
task/
├── task.toml
├── instruction.md
├── environment/
├── round_1/
│   ├── instruction.md
│   ├── solution/
│   └── tests/
├── round_2/
└── round_N/
```

All rounds run in the same workspace unless the run is explicitly resumed or expanded from a saved snapshot. Later rounds inherit the code and durable environment state created by earlier rounds.

## Execution Window

Every run has a closed execution window:

```text
start_round..min(max_round, task.num_rounds)
```

Before the window:

- A fresh `--start-round N` run fast-forwards prior rounds by applying Oracle solutions inside the same environment.
- A `--resume-trial --resume-round N` run restores the previous trial's boundary state and copies historical verifier results.

Inside the window, each round follows the same sequence:

1. Read `round_N/instruction.md`.
2. Call `agent.run_round(...)`.
3. Run the round verifier from `round_N/tests/`.
4. Archive round-specific verifier artifacts.
5. Capture snapshot and agent continuation state if policy allows.

After the window, future rounds are not executed and do not produce round artifacts.

## Scoring

The final reward is the mean over the aggregate window, not just the last round reward.

Default aggregate windows:

- fresh run: executed window
- in-place resume: round 1 through the new right boundary
- fanout child: the child trial's responsible window

Explicit `--multiround-aggregate-start-round` and `--multiround-aggregate-end-round` override the default. The resolved window is written to `result.json`.

## Resume

Resume restores a round boundary, not a live shell process. It uses:

- environment snapshot metadata from `state/round_N/`
- agent continuation state from `agent/session_snapshots/` or `agent/runtime_snapshots/`
- copied historical verifier results for rounds before the resume point

Before mutating a target trial directory, CLI preflight checks the source state according to `multiround_resume_preflight_policy`.

Resume output modes:

- default in-place resume backs up the source trial under `__resumed_<timestamp>/`, cleans artifacts from the resume point onward, and continues in the original trial directory
- `--output-jobs-dir` writes a new job from the source trial without mutating the source trial
- `--no-resume-backup` keeps the in-place write behavior but skips the automatic backup

## Roundwise Fanout

With `-k > 1`, eligible multiturn jobs can use roundwise attempt selection:

- round 1 creates a frontier of attempts
- successful parents are selected according to `multiround_continue_successes_per_round`
- child trials start at the next round from the selected parent snapshot
- child names include the parent attempt and lineage token
- snapshot state for unselected frontier trials is pruned after parent selection

The scheduler waits for the full current frontier before selecting parents. It does not cancel sibling attempts after the first success.

## Upstream Compatibility Principle

The overlay follows upstream Harbor architecture where possible:

- `Task` owns task parsing and validation.
- `Job` owns trial expansion and attempt selection.
- `Trial` owns shared lifecycle and environment setup.
- `SingleStepTrial` owns the single-step and multiturn execution loop.
- Agents expose multiturn behavior through `run_round()` rather than replacing the base agent contract.

This keeps future upstream merges concentrated around a small set of extension points.
