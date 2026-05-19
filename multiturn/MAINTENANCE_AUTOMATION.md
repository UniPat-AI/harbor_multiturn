# Maintenance Automation Boundary

This document defines what the local multiturn maintenance flow can automate and
what must stay under human review. The goal is repeatability without turning
upstream Harbor updates into unattended semantic merges.

## Automated

Automation may do the following without changing stable branches:

- Fetch upstream Harbor into local tracking refs.
- Run deterministic gates:
  - Harbor multiturn unit subset.
  - Parent `tests/manifests/multiturn_core.txt`.
  - Shell syntax checks and whitespace checks.
- Validate local Daytona environment variables without printing secrets.
- Launch credentialed MT@4/SR jobs when the operator has explicitly provided
  local credentials and accepted the cost/runtime.
- Summarize changed files, conflicts, failing tests, and likely conflict zones
  after the maintainer starts a manual merge.

Automation must not create or maintain upstream-merge branches. It also should
not decide whether an upstream behavior change is accepted.

## Human Review

A maintainer must decide:

- Whether an upstream Harbor behavioral change should replace local multiturn
  behavior or be wrapped by a multiturn compatibility hook.
- Whether to run an optional manual detached-worktree merge rehearsal.
- How to resolve conflicts in lifecycle, CLI, task parsing, environment setup,
  agent continuation state, verifier output, and job fanout.
- Whether credentialed MT@4/SR results are acceptable for release-like use.
- Whether to commit and push `multiturn/main`.
- Whether to advance the parent repository's `harbor/` gitlink.
- Whether local credentials, run logs, or generated outputs accidentally entered
  a staged diff.

Agents may propose conflict resolutions and edit code, tests, and docs, but they
should leave merge commits and pushes to explicit maintainer approval.

## Daytona Checks

Daytona credentials are local-only. The committed repo may document required
variable names, but must not contain real keys.

Before launching a Daytona-backed credentialed run:

```bash
bash tests/run_multiturn_maintenance.sh --daytona-preflight --unit-only
```

This calls `tests/check_daytona_multiturn_env.sh`, which verifies:

- `HARBOR_ENV=daytona`
- `DAYTONA_API_KEY` is set but never printed
- `HARBOR_ENV_KWARGS` includes finite `auto_stop_interval_mins`
- `HARBOR_ENV_KWARGS` includes `auto_delete_interval_mins`

## Agent Collaboration Loop

For upstream updates, the recommended human+agent loop is:

1. Human decides whether to rehearse the upstream merge in a manual detached worktree.
2. Agent reports conflicts and changed upstream areas.
3. Agent resolves mechanical conflicts and runs deterministic tests.
4. Human reviews semantic conflict zones and credentialed test need.
5. Agent runs MT@4/SR or Daytona preflight only after credentials are already
   present locally.
6. Human approves final commit/push and parent gitlink update.
