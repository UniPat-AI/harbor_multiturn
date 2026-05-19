# Maintenance Automation Boundary

This document defines the local multiturn maintenance split between repeatable
automation and human review. The goal is a predictable upstream update loop for
`multiturn/main`.

## Automated Gate

Automation owns repeatable checks and summaries:

- Fetch upstream Harbor into local tracking refs.
- Run deterministic gates:
  - Harbor multiturn unit subset.
  - Parent `tests/manifests/multiturn_core.txt`.
  - Shell syntax checks and whitespace checks.
- Validate local Daytona environment variables with redacted credential output.
- Launch credentialed MT@4/SR jobs when the operator has explicitly provided
  local credentials and accepted the cost/runtime.
- Summarize changed files, conflicts, failing tests, and likely conflict zones
  after the maintainer starts a manual merge.

## Human Review

A maintainer owns semantic decisions and repository mutation:

- Upstream Harbor behavioral changes that should replace local multiturn
  behavior or be wrapped by a multiturn compatibility hook.
- Optional manual detached-worktree merge rehearsal.
- Conflict resolution in lifecycle, CLI, task parsing, environment setup,
  agent continuation state, verifier output, and job fanout.
- Credentialed MT@4/SR release-like acceptance.
- Commit and push approval for `multiturn/main`.
- Parent repository `harbor/` gitlink updates.
- Final staged diff review for credentials, run logs, and generated outputs.

Agents may propose conflict resolutions and edit code, tests, and docs. Merge
commits and pushes happen after explicit maintainer approval.

## Daytona Checks

Daytona credentials live in local environment configuration. The committed repo
documents variable names and lifecycle settings.

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
