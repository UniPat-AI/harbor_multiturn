# Multiturn Support Matrix

## Task Format

A task enters the multiturn path when `task.toml` contains:

```toml
[metadata.multiround]
num_rounds = 3

[[metadata.multiround.rounds]]
round = 1
change_types = ["extension"]
```

Current rules:

- `num_rounds >= 1` is accepted. `num_rounds = 1` is treated as an in-progress multiturn task and still uses round-aware validation.
- Round directories must be exactly `round_1..round_N`.
- Each round must provide `instruction.md`, an OS-compatible `solution/solve.*`, and an OS-compatible `tests/test.*`.
- `change_types` is the preferred field. Legacy `change_type = "extension"` is normalized into `change_types = ["extension"]`.
- Allowed change types are `extension`, `correction`, and `conflict`.
- `round_N/tests/` is the complete verifier suite for that round, not a patch over previous tests.

Top-level `instruction.md` remains required for compatibility with Harbor task discovery. Actual round prompts are read from `round_N/instruction.md`.

## Agents

Multiturn tasks require an agent-specific `run_round()` implementation. Supported agents:

| Agent | Status | Notes |
| --- | --- | --- |
| `oracle` | Supported | Uploads and runs `round_N/solution/` per round. |
| `claude-code` | Supported | Keeps per-round output, Claude session snapshots, and same-agent continuation state. |
| `terminus-2` | Supported | Keeps chat/runtime trajectory state and tmux recovery metadata. |

Agents that only implement Harbor's single-step `run()` path are rejected before a multiturn job directory is created.

## CLI Options

Supported round controls:

- `--start-round N`
- `--max-round M`
- `--resume-trial PATH`
- `--resume-round N`
- `--multiround-state-cache-policy {off,success,all}`
- `--multiround-resume-preflight-policy {off,snapshot,strict}`
- `--multiround-continue-successes-per-round K`
- `--multiround-aggregate-start-round N`
- `--multiround-aggregate-end-round M`

Internal resume fields may appear in `config.json`:

- `multiround_resume_state_image`
- `multiround_resume_state_archive`
- `multiround_resume_state_snapshot_id`

They are written by resume and fanout plumbing, not by users directly.

## Environment State

Round snapshots are environment-dependent. The local Docker environment supports snapshot capture and restore through per-round Docker image archives. Snapshot metadata is written under:

```text
trial/state/round_N/snapshot.json
trial/state/round_N/snapshot-image.tar
trial/state/snapshot.json
trial/state/snapshot-image.tar
```

`state/snapshot.*` is the latest successful alias. Resume and roundwise fanout use the concrete `round_N` snapshot as the durable restore point.

## Unsupported Combinations

The CLI rejects combinations that would make the runtime semantics ambiguous:

- multiturn task with unsupported agent
- `--disable-verification` on a multiturn task
- `--resume-trial` without `--resume-round`
- `--resume-round < 2`
- `--resume-trial` with `--start-round`
- `--start-round > --max-round`
- `--resume-trial` on dataset or multi-task jobs
- roundwise fanout that requires snapshots while `multiround_state_cache_policy=off`

Cross-agent resume is intentionally narrow. Oracle-to-agent handoff is supported for the tested paths; non-Oracle cross-agent handoff must pass the current CLI validation and resume preflight checks.
