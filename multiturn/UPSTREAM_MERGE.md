# Upstream Merge Workflow

## Branch Roles

Use these roles consistently:

| Ref | Role |
| --- | --- |
| `origin/main` | Latest upstream Harbor tracking ref. Treat this as read-only input from Harbor. |
| `multi_turn_support` | Stable pre-upstream-merge multiturn branch. This is the rollback baseline and the remote default branch for users who need the known-good old multiturn code. |
| `multiturn/main` | Maintained integration branch for daily work: latest upstream Harbor plus the multiturn overlay. New upstream Harbor releases are merged here. |
| parent repo `master` | Records the selected `harbor/` submodule commit and owns the `multiturnpp` docs/tests. |
| `backup/multiturn-main-before-upstream-*` | Per-merge rollback branches. |
| `backup/multiturn-main-before-upstream-*` tags | Per-merge immutable rollback labels. |

The intended long-term shape is:

```text
origin/main          # moving upstream Harbor input
multi_turn_support   # stable multiturn fallback
multiturn/main       # moving integration branch: merge origin/main, repair overlay, test, then update parent gitlink
```

Use `multiturn/main` for all future Harbor updates. Keep `multi_turn_support`
available as the stable rollback and bisection baseline.

## Optional Manual Rehearsal

Before mutating `multiturn/main`, the maintainer may manually rehearse the
upstream merge in a detached worktree:

```bash
cd /home/shenhaiyang/Source/swebenchpp/multiturnpp
git -C harbor fetch origin --prune
git -C harbor worktree add --detach data/maintenance/upstream_merge_dry_run/harbor multiturn/main
cd data/maintenance/upstream_merge_dry_run/harbor
git merge --no-commit --no-ff origin/main
```

If the rehearsal merge is useful, inspect conflicts and run the deterministic
gate from the parent directory with `HARBOR_DIR` pointed at the detached
worktree:

```bash
cd /home/shenhaiyang/Source/swebenchpp/multiturnpp
HARBOR_DIR=$PWD/data/maintenance/upstream_merge_dry_run/harbor \
  bash tests/run_multiturn_maintenance.sh
```

Discard the rehearsal after inspection:

```bash
git -C harbor worktree remove --force data/maintenance/upstream_merge_dry_run/harbor
```

The maintained branch set remains `origin/main`, `multi_turn_support`, and
`multiturn/main`.

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

Recommended parent-side finish:

```bash
cd ..
git add harbor TEST.md tests/
git commit -m "Update Harbor multiturn integration"
```

Only stage the submodule gitlink and the multiturn docs/tests you intentionally changed.

## Conflict Policy

Prefer upstream Harbor structure, then reapply the local multiturn hooks.

Common conflict zones:

| Area | Merge rule |
| --- | --- |
| `src/harbor/models/task/task.py` | Keep upstream task features such as extra instructions, then preserve multiturn shape and round metadata validation. |
| `src/harbor/cli/jobs.py` | Keep upstream CLI options and queue setup, then preserve multiturn validation and resume parameter plumbing. |
| `src/harbor/multiround/` | Keep local pure helper modules here when possible so upstream Harbor file conflicts stay small. |
| `src/harbor/job.py` | Keep upstream `TrialQueue` behavior, then preserve roundwise attempt selection. |
| `src/harbor/trial/trial.py` | Keep upstream lifecycle setup, injected skills, hooks, and environment creation, then preserve resume snapshot resolution. |
| `src/harbor/trial/single_step.py` | Preserve upstream single-step behavior for normal tasks; keep multiturn as a branch for `task.is_multiround`. |
| `src/harbor/environments/docker/docker.py` | Keep upstream Docker platform/mount changes, then preserve snapshot restore/capture. |
| agent files | Keep upstream install/run APIs, then preserve `run_round()` and continuation state. |

## Conflict Reduction Rules

- Keep local documentation in `multiturn/`.
- Keep deterministic black-box maintenance tests in the parent `tests/` harness.
- Keep new local runtime behavior behind existing Harbor extension points.
- Move pure multiturn planning and selection logic into `src/harbor/multiround/`
  when it is independent from CLI or Trial side effects.
- Preserve upstream formatting in files that upstream changes frequently.
- Use merge commits for `multiturn/main`.
- Separate commits by purpose: upstream merge, local conflict repair, docs/tests.
- Keep the generated `mleval` fixture under the parent `tests/tasks/` tree.
- Prefer adding local multiturn docs under `harbor/multiturn/`; update upstream documentation only when the upstream project contract changes.

## Recurring Maintenance Checklist

For each upstream Harbor update:

1. Optionally rehearse the merge in a detached worktree and inspect the resulting tree.
2. Backup `multiturn/main` with a branch and tag before the real merge.
3. Merge `origin/main` into `multiturn/main`.
4. Resolve conflicts by keeping upstream structure first, then reapplying multiturn hooks.
5. Run `bash tests/run_multiturn_maintenance.sh`.
6. If agent/session/environment code changed, also run the credentialed Terminus-2 generated-task MT@4/SR commands from `TESTING.md`.
7. Commit Harbor, update the parent submodule pointer, and commit parent docs/tests.

## Rollback

To inspect the pre-merge state:

```bash
git log --oneline --decorate --graph --first-parent multiturn/main
git switch backup/multiturn-main-before-upstream-<stamp>
```

To return `multiturn/main` to a known backup, create a new branch or use a
non-destructive revert.
