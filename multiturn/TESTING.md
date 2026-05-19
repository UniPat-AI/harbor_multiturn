# Multiturn Maintenance Testing

## Canonical Three-Round Task

The continuous regression task is:

```text
../tests/tasks/theme_d10_w10_ml_ai_mlops_forensics_analysis/
```

It has exactly three rounds:

| Round | Change type | Purpose |
| --- | --- | --- |
| 1 | `extension` | Establish the baseline implementation. |
| 2 | `correction` | Change prior behavior and verify updated expectations. |
| 3 | `conflict` | Replace conflicting assumptions and verify the final state. |

This task is the fixture for the parent black-box manifest `tests/manifests/multiturn_core.txt`.

## Primary Maintenance Command

From the parent `multiturnpp` directory:

```bash
bash tests/run_multiturn_maintenance.sh
```

The script runs:

1. Harbor unit tests that cover multiturn parsing, resume, snapshots, trajectory merge, agent setup, and selected upstream integration points.
2. The black-box `multiturn_core` manifest against the three-round task.

Useful variants:

```bash
bash tests/run_multiturn_maintenance.sh --unit-only
bash tests/run_multiturn_maintenance.sh --blackbox-only
bash tests/run_multiturn_maintenance.sh --cleanup
```

`--cleanup` forwards cleanup to the black-box harness so per-case job directories are removed after each case.

## Unit Coverage

The maintenance script runs:

```text
harbor/tests/unit/multiround/
harbor/tests/unit/test_task_relative_path.py
harbor/tests/unit/models/test_job_lock.py
harbor/tests/unit/environments/test_docker.py
harbor/tests/unit/agents/installed/test_claude_code_resume_context.py
harbor/tests/unit/agents/installed/test_setup_retry.py
harbor/tests/unit/agents/terminus_2/test_resume_context_equivalence.py
harbor/tests/unit/agents/terminus_2/test_session_recovery.py
harbor/tests/unit/agents/terminus_2/test_tmux_current_path.py
harbor/tests/unit/agents/terminus_2/test_tmux_recording_fallback.py
harbor/tests/unit/agents/terminus_2/test_tmux_session.py
```

Run the full Harbor unit suite when an upstream merge touches shared lifecycle, environment, task config, CLI, queue, or verifier behavior:

```bash
cd harbor
uv run python -m pytest tests/unit
```

## Black-Box Coverage

`tests/manifests/multiturn_core.txt` covers:

- full three-round Oracle run
- `--max-round`
- `--start-round`
- combined `--start-round` and `--max-round`
- in-place resume from round 2
- unsupported-agent preflight
- CLI parameter validation
- reward aggregation
- single-chain versus roundwise fanout control flow
- selected-parent snapshot retention and unselected frontier snapshot pruning

This manifest intentionally uses Oracle so the maintenance loop is deterministic and does not require external model credentials.

## Merge Gate

After every upstream Harbor merge into `multiturn/main`, run:

```bash
bash tests/run_multiturn_maintenance.sh
```

If the merge changed broad upstream behavior, also run:

```bash
cd harbor
uv run python -m pytest tests/unit
cd ..
bash tests/run.sh --manifest quick
```

Agent E2E manifests remain credentialed and heavier; run them before release-like changes or when Claude Code / Terminus-2 resume behavior changes.
