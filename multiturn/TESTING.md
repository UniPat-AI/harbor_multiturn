# Multiturn Maintenance Testing

## Canonical Three-Round Tasks

The maintenance gate keeps two local three-round tasks:

| Task | Role |
| --- | --- |
| `../tests/tasks/theme_d10_w10_ml_ai_mlops_forensics_analysis/` | Compact deterministic task used by the broad Oracle matrix: full run, windows, resume, validation, aggregation, and fanout control flow. |
| `../tests/tasks/theme_d10_w11_ml_ai_mlops_automation_scripting/` | Clean copy of the generated `mleval` task from `terminal_bench_tasks_multiturn_light_20260515`; used for continuous MT@4 and SR regression on a realistic generated task. |

The generated `mleval` task has exactly three rounds:

| Round | Change type | Purpose |
| --- | --- | --- |
| 1 | `extension` | Establish the `mleval` CLI and registry baseline. |
| 2 | `extension`, `correction` | Add archive/CSV behavior while changing prior listing semantics. |
| 3 | `extension`, `conflict` | Add sample statistics/significance behavior and resolve conflicting ranking semantics. |

Both tasks are fixtures for the parent black-box manifest `tests/manifests/multiturn_core.txt`.

## Primary Maintenance Command

From the parent `multiturnpp` directory:

```bash
bash tests/run_multiturn_maintenance.sh
```

The script runs:

1. Harbor unit tests that cover multiturn parsing, resume, snapshots, trajectory merge, agent setup, and selected upstream integration points.
2. The black-box `multiturn_core` manifest against the compact task plus the generated `mleval` MT@4/SR case.

The parent repository also carries `.github/workflows/multiturn-maintenance.yml`, which runs the unit gate and the deterministic black-box gate when `multiturnpp/**` changes. The local multiturn gate is intentionally kept out of Harbor's upstream `.github/workflows` so future upstream merges do not create workflow conflicts.

Useful variants:

```bash
bash tests/run_multiturn_maintenance.sh --unit-only
bash tests/run_multiturn_maintenance.sh --blackbox-only
bash tests/run_multiturn_maintenance.sh --cleanup
bash tests/run_multiturn_maintenance.sh --daytona-preflight --unit-only
```

`--cleanup` forwards cleanup to the black-box harness so per-case job directories are removed after each case.
`--daytona-preflight` validates local Daytona variables without printing
`DAYTONA_API_KEY`; it does not contact Daytona or launch a sandbox.

## Unit Coverage

The maintenance script runs:

```text
harbor/tests/unit/multiround/
harbor/tests/unit/cli/
harbor/tests/unit/test_task_relative_path.py
harbor/tests/unit/models/test_job_lock.py
harbor/tests/unit/environments/test_docker.py
harbor/tests/unit/agents/installed/test_claude_code_resume_context.py
harbor/tests/unit/agents/installed/test_setup_retry.py
harbor/tests/unit/agents/terminus_2/test_resume_context_equivalence.py
harbor/tests/unit/agents/terminus_2/test_session_recovery.py
harbor/tests/unit/agents/terminus_2/test_terminus_2_mcp.py
harbor/tests/unit/agents/terminus_2/test_terminus_2_temperature.py
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
- generated `mleval` MT@4: `-k 4`, one successful parent continued per round
- generated `mleval` SR: `--start-round N --max-round N` for rounds 1, 2, and 3

This manifest intentionally uses Oracle so the maintenance loop is deterministic and does not require external model credentials.

To run only the generated task regression:

```bash
bash tests/run.sh --test 28 --cleanup
```

## Credentialed Terminus-2 Validation

The deterministic gate above proves Harbor's multiturn plumbing without API credentials. Before release-like updates, run the generated task with the same Terminus-2 shape used during task production.

Use the committed clean fixture so runtime output lands under an ignored `harbor_jobs/` directory:

```bash
cd /home/shenhaiyang/Source/swebenchpp/multiturnpp
export TASKS_DIR=/home/shenhaiyang/Source/swebenchpp/multiturnpp/tests/tasks
export TASK_NAME=theme_d10_w11_ml_ai_mlops_automation_scripting
```

For Daytona-backed smoke runs, configure the sandbox locally before launching:

```bash
export HARBOR_ENV=daytona
export HARBOR_ENV_KWARGS=network_block_all=false,auto_stop_interval_mins=120,auto_delete_interval_mins=240
export DAYTONA_API_URL=https://app.daytona.io/api
export DAYTONA_TARGET=us
# DAYTONA_API_KEY must already be set in the local shell/profile/task .env.
bash tests/check_daytona_multiturn_env.sh
```

Daytona currently validates fresh-run plumbing. Do not treat it as equivalent
to Docker for resume/fanout snapshot coverage until `DaytonaEnvironment`
implements Harbor's per-round `capture_state_snapshot()` contract.

MT@4:

```bash
tmux new-session -d -s mt4_${TASK_NAME}_$(date +%Y%m%d) \
"cd /home/shenhaiyang/Source/swebenchpp/multiturnpp && \
export TASKS_DIR=/home/shenhaiyang/Source/swebenchpp/multiturnpp/tests/tasks && \
export TASK_NAME=theme_d10_w11_ml_ai_mlops_automation_scripting && \
export JOBS_DIR_OVERRIDE=\"\$TASKS_DIR/\$TASK_NAME/harbor_jobs/mt4_opus47_medium_$(date +%Y%m%d)\" && \
export AGENT_TYPE=terminus-2 AGENT_MODEL=openai/claude-opus-4-7 API_PROVIDER=mindra \
AGENT_ATTEMPTS=4 HARBOR_N_CONCURRENT=4 \
MULTIROUND_CONTINUE_SUCCESSES_PER_ROUND=1 AGENT_CONTINUE_SUCCESSES_PER_ROUND=1 \
TASK_TIMEOUT=43200 && \
export AGENT_KWARGS='{\"reasoning_effort\":\"medium\"}' && \
autonomous/lightversion/3_harbor_test_multiple.sh \"\$TASKS_DIR\" agent -t \"\$TASK_NAME\" -j 1 -T 43200 -- --force --agent-continue-successes-per-round 1"
```

SR for a target round:

```bash
export JOBS_DIR_OVERRIDE="$TASKS_DIR/$TASK_NAME/harbor_jobs/sr_r3_opus47_medium_$(date +%Y%m%d)"
export AGENT_TYPE=terminus-2 AGENT_MODEL=openai/claude-opus-4-7 API_PROVIDER=mindra
export AGENT_ATTEMPTS=4 HARBOR_N_CONCURRENT=4 TASK_TIMEOUT=43200
export AGENT_KWARGS='{"reasoning_effort":"medium"}'

autonomous/lightversion/3_harbor_test_multiple.sh "$TASKS_DIR" agent -t "$TASK_NAME" -j 1 -T 43200 -- \
  --force --start-round 3 --max-round 3
```

Repeat SR with `--start-round 1 --max-round 1`, `--start-round 2 --max-round 2`, and `--start-round 3 --max-round 3`.

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
