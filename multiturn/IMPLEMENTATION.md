# Multiturn Implementation Map

## Task Parsing

Files:

- `src/harbor/models/task/paths.py`
- `src/harbor/models/task/task.py`

Responsibilities:

- Discover `round_N` directories.
- Resolve OS-compatible `round_N/solution/solve.*` and `round_N/tests/test.*`.
- Validate that declared round metadata matches the filesystem.
- Normalize `change_type` and `change_types`.
- Exclude volatile run artifacts from task checksums.
- Append Harbor runtime extra instruction files to normal and round instructions.

## CLI and Config

Files:

- `src/harbor/cli/jobs.py`
- `src/harbor/multiround/resume_plan.py`
- `src/harbor/models/trial/config.py`
- `src/harbor/models/trial/paths.py`

Responsibilities:

- Expose round window, resume, aggregate window, state cache, and fanout parameters.
- Reject invalid multiturn combinations before jobs start.
- Plan resume output mode, inferred jobs directory, job name, and backup naming without filesystem side effects.
- Print the resume dry-run report before any backup, cleanup, or job creation.
- Resolve in-place resume targets and snapshot source metadata.
- Persist multiturn parameters to `config.json`.

## Job Expansion

File:

- `src/harbor/job.py`

Responsibilities:

- Validate that multiturn tasks use agents with explicit round support.
- Use upstream `TrialQueue` scheduling for normal jobs.
- Switch eligible `-k > 1` multiturn runs into roundwise frontier/child expansion.
- Pass parent snapshot and lineage metadata into child trial configs.

## Trial Lifecycle

Files:

- `src/harbor/trial/trial.py`
- `src/harbor/trial/single_step.py`

Responsibilities:

- Build the agent, environment, verifier, artifact handler, and injected skills using upstream lifecycle hooks.
- Resolve resume snapshot fields before environment startup.
- Detect whether the environment implements snapshot capture.
- Keep compatibility aliases used by local multiturn helpers.
- Branch `SingleStepTrial._run()` into the multiturn loop when `task.is_multiround`.
- Archive per-round verifier artifacts and aggregate final results.
- Merge same-agent resume trajectory data without deleting valid current-round repeated steps.

## Agents

Files:

- `src/harbor/agents/base.py`
- `src/harbor/agents/oracle.py`
- `src/harbor/agents/installed/base.py`
- `src/harbor/agents/installed/claude_code.py`
- `src/harbor/agents/terminus_2/terminus_2.py`
- `src/harbor/agents/terminus_2/tmux_session.py`

Responsibilities:

- `BaseAgent.run_round()` defaults to `run()` for compatibility, but multiturn jobs only accept agents with concrete round support.
- Oracle applies the round solution in the active environment.
- Claude Code stores round output files, sessions, session snapshots, and resume metadata.
- Terminus-2 stores chat/runtime state, trajectory metadata, tmux recovery data, and round-aware trajectory entries.
- Installed-agent setup includes retry handling for transient setup failures.

## Environment Snapshots

Files:

- `src/harbor/environments/base.py`
- `src/harbor/environments/docker/docker.py`

Responsibilities:

- Define the environment snapshot capability.
- Restore Docker environments from snapshot image/archive metadata.
- Capture per-round Docker image snapshots.
- Prepare logs and writable mounts for host access during stop/download.

## Output Contract

Multiturn trials write:

```text
trial/
├── result.json
├── trial.log
├── verifier/
│   ├── multiround_results.json
│   ├── round_N_reward.txt
│   ├── round_N_reward.json
│   ├── round_N_test-stdout.txt
│   ├── round_N_test-stderr.txt
│   └── round_N_test-exit-code.txt
├── state/
│   ├── snapshot.json
│   └── round_N/snapshot.json
└── agent/
    ├── trajectory.json
    ├── session_snapshots/
    ├── runtime_snapshots/
    └── pre_resume_trajectories/
```

The parent `multiturnpp/TEST.md` contains the full black-box artifact matrix.

## Change Discipline

When changing implementation:

1. Prefer upstream abstractions over new parallel paths.
2. Keep round-specific behavior in the existing extension points above.
3. Add or update unit tests in `tests/unit/multiround/` or the nearest affected upstream test file.
4. Add black-box coverage in the parent `tests/cases/` only when observable CLI behavior or artifact shape changes.
5. Run `bash ../tests/run_multiturn_maintenance.sh` from this directory's parent before considering the change stable.
