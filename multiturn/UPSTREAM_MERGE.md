# Upstream Merge Workflow

## Branch Roles

Use these roles consistently:

| Ref | Role |
| --- | --- |
| `origin/main` | Upstream Harbor tracking ref. Do not edit directly. |
| `multi_turn_support` | Original multiturn implementation baseline. Keep as rollback context. |
| `multiturn/main` | Maintained integration branch for daily work and upstream merges. |
| `backup/multiturn-main-before-upstream-*` | Per-merge rollback branches. |
| `backup/multiturn-main-before-upstream-*` tags | Per-merge immutable rollback labels. |

## Standard Update

From `harbor/`:

```bash
git fetch origin --prune
git switch multiturn/main

stamp="$(date +%Y%m%d_%H%M%S)"
git branch "backup/multiturn-main-before-upstream-${stamp}"
git tag "backup/multiturn-main-before-upstream-${stamp}"

git merge origin/main
# resolve conflicts
cd ..
bash tests/run_multiturn_maintenance.sh
cd harbor
git commit
```

If the parent repository records `harbor` as a submodule, update the parent gitlink after the Harbor commit is stable.

## Conflict Policy

Prefer upstream Harbor structure, then reapply the local multiturn hooks. Do not revive removed upstream architecture just because older multiturn code depended on it.

Common conflict zones:

| Area | Merge rule |
| --- | --- |
| `src/harbor/models/task/task.py` | Keep upstream task features such as extra instructions, then preserve multiturn shape and round metadata validation. |
| `src/harbor/cli/jobs.py` | Keep upstream CLI options and queue setup, then preserve multiturn validation and resume parameter plumbing. |
| `src/harbor/job.py` | Keep upstream `TrialQueue` behavior, then preserve roundwise attempt selection. |
| `src/harbor/trial/trial.py` | Keep upstream lifecycle setup, injected skills, hooks, and environment creation, then preserve resume snapshot resolution. |
| `src/harbor/trial/single_step.py` | Preserve upstream single-step behavior for normal tasks; keep multiturn as a branch for `task.is_multiround`. |
| `src/harbor/environments/docker/docker.py` | Keep upstream Docker platform/mount changes, then preserve snapshot restore/capture. |
| agent files | Keep upstream install/run APIs, then preserve `run_round()` and continuation state. |

## Conflict Reduction Rules

- Keep local documentation in `multiturn/`; do not edit upstream `README.md` for local behavior.
- Keep deterministic black-box maintenance tests in the parent `tests/` harness; do not duplicate them inside upstream Harbor examples.
- Keep new local runtime behavior behind existing Harbor extension points.
- Avoid broad formatting rewrites in files that upstream changes frequently.
- Use merge commits, not rebases, for `multiturn/main`.
- Separate commits by purpose: upstream merge, local conflict repair, docs/tests.

## Rollback

To inspect the pre-merge state:

```bash
git log --oneline --decorate --graph --first-parent multiturn/main
git switch backup/multiturn-main-before-upstream-<stamp>
```

To return `multiturn/main` to a known backup, create a new branch or use a non-destructive revert. Avoid `git reset --hard` unless the caller explicitly requests it.
