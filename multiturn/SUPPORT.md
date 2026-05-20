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
- Multiturn tasks require Harbor shared verifier mode. `[verifier].environment_mode = "separate"` and `[verifier.environment]` are rejected because round verification targets the live agent workspace.

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
- `--resume-mode {inplace-backup,copy,inplace-no-backup}`
- `--output-jobs-dir DIR`
- `--no-resume-backup`
- `--resume-dry-run`
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

`state/snapshot.*` is the latest saved alias. Resume and roundwise fanout use
the concrete `round_N` snapshot as the durable restore point. In fresh
roundwise runs with the default `success` policy, snapshots are selected-only:
Harbor captures a snapshot only for selected successful parents, including the
final round. The default `latest` retention policy keeps only the latest
selected round snapshot. Use `selected` to keep the selected chain, or `all` to
avoid retention pruning when debugging.

Daytona Direct implements the same Harbor per-round state contract. Its default
`multiround_state_mode=auto` resolves to `pause_fork`: Harbor saves a local
rootfs archive for the selected sandbox at the round boundary, records the
sandbox as the next round's fork source, deletes unselected siblings, then
tries to fork the selected source when a child trial resumes. If the Daytona
service does not expose sandbox fork or dynamic sandbox snapshot endpoints,
Harbor restores the child from the local archive. `multiround_state_mode=snapshot`
uses Daytona snapshots when available and otherwise falls back to the same
local archive restore path. `multiround_state_mode=archive` uses only the local
archive path.

Daytona state metadata is written to the same `state/round_N/snapshot.json`
contract. Daytona entries use provider-specific references such as
`daytona-fork-source:<sandbox_id>`, `daytona-snapshot:<snapshot_name>`, or
`daytona-archive:<snapshot_id>`. Archive fallback writes `snapshot-image.tar`
and `archive_path` just like the Docker state contract. DinD/Compose-backed
Daytona environments are not a multiturn state backend; Compose-heavy
multiturn coverage stays on local Docker.

## Unsupported Combinations

The CLI rejects combinations that would make the runtime semantics ambiguous:

- multiturn task with unsupported agent
- `--disable-verification` on a multiturn task
- `--resume-trial` without `--resume-round`
- `--resume-mode` without `--resume-trial`
- `--resume-mode copy` without `--output-jobs-dir`
- `--resume-dry-run` without `--resume-trial`
- `--resume-round < 2`
- `--resume-trial` with `--start-round`
- `--output-jobs-dir` without `--resume-trial`
- `--start-round > --max-round`
- `--resume-trial` on dataset or multi-task jobs
- roundwise child expansion that requires snapshots while `multiround_state_cache_policy=off`
- multiturn task with Harbor separate verifier mode

Cross-agent resume is intentionally narrow. Oracle-to-agent handoff is supported for the tested paths; non-Oracle cross-agent handoff must pass the current CLI validation and resume preflight checks.
