# Maintenance Automation Boundary

This document defines what the local multiturn maintenance flow can automate and
what must stay under human review. The goal is repeatability without turning
upstream Harbor updates into unattended semantic merges.

## Automated

Automation may do the following without changing stable branches:

- Fetch upstream Harbor into local tracking refs.
- Prepare a dry-run merge in a fixed detached worktree under
  `data/maintenance/upstream_merge_dry_run/harbor`.
- Run deterministic gates:
  - Harbor multiturn unit subset.
  - Parent `tests/manifests/multiturn_core.txt`.
  - Shell syntax checks and whitespace checks.
- Validate local Daytona environment variables without printing secrets.
- Launch credentialed MT@4/SR jobs when the operator has explicitly provided
  local credentials and accepted the cost/runtime.
- Summarize changed files, conflicts, failing tests, and likely conflict zones.

Automation must be idempotent where possible. Reuse the fixed dry-run worktree
instead of creating any temporary branches for upstream updates.

## Human Review

A maintainer must decide:

- Whether an upstream Harbor behavioral change should replace local multiturn
  behavior or be wrapped by a multiturn compatibility hook.
- How to resolve conflicts in lifecycle, CLI, task parsing, environment setup,
  agent continuation state, verifier output, and job fanout.
- Whether credentialed MT@4/SR results are acceptable for release-like use.
- Whether to commit and push `multiturn/main`.
- Whether to advance the parent repository's `harbor/` gitlink.
- Whether local credentials, run logs, or generated outputs accidentally entered
  a staged diff.

Agents may propose conflict resolutions and edit code, tests, and docs, but they
should leave merge commits and pushes to explicit maintainer approval.

## Fixed Dry-Run Worktree

Use the same detached worktree path for every upstream merge rehearsal:

```bash
cd /home/shenhaiyang/Source/swebenchpp/multiturnpp
bash tests/upstream_merge_dry_run.sh
```

Default refs:

```text
BASE_BRANCH=multiturn/main
UPSTREAM_REMOTE=origin
UPSTREAM_BRANCH=main
DRY_RUN_WORKTREE=data/maintenance/upstream_merge_dry_run/harbor
```

The script creates no branch. It checks out `multiturn/main` into the detached
worktree, performs `origin/main --no-commit --no-ff` there, and leaves the merge
uncommitted so a maintainer or agent can inspect the exact combined tree. After
inspection, discard the worktree with `git worktree remove --force`.

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

1. Human asks agent to run the fixed dry-run merge.
2. Agent reports conflicts and changed upstream areas.
3. Agent resolves mechanical conflicts and runs deterministic tests.
4. Human reviews semantic conflict zones and credentialed test need.
5. Agent runs MT@4/SR or Daytona preflight only after credentials are already
   present locally.
6. Human approves final commit/push and parent gitlink update.
