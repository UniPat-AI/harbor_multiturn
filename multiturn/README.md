# Harbor Multiturn Overlay

This directory documents the local multiturnpp extension carried by the Harbor fork. It is intentionally separate from Harbor's upstream `README.md` and documentation site so upstream merges stay focused on code conflicts rather than local documentation churn.

## Document Map

| Document | Purpose |
| --- | --- |
| `SUPPORT.md` | Supported task shape, agents, CLI options, and unsupported combinations. |
| `DESIGN.md` | Runtime semantics for round execution, resume, snapshots, scoring, and fanout. |
| `IMPLEMENTATION.md` | Code map for the multiturn implementation inside current Harbor. |
| `TESTING.md` | The continuous maintenance test loop based on the 3-round local task. |
| `UPSTREAM_MERGE.md` | Branch model and merge workflow for keeping current with upstream Harbor. |
| `ROADMAP.md` | Backward-compatible optimization plan for lowering upstream merge friction and expanding coverage. |

## Branch Model

The maintained integration branch is `multiturn/main`.

```text
origin/main          # upstream Harbor, read-only local tracking ref
multi_turn_support   # stable pre-merge multiturn baseline and rollback branch
multiturn/main       # maintained branch: upstream Harbor plus multiturn overlay
```

Use merge commits to bring `origin/main` into `multiturn/main`. Do not rebase `multiturn/main`; the merge boundary is useful when auditing which upstream update introduced a conflict or regression.

The parent `multiturnpp` repository records the chosen Harbor commit as the `harbor/` submodule pointer. After `multiturn/main` is stable, update that parent gitlink and commit the matching docs/tests there.

## Maintenance Loop

From the parent `multiturnpp` directory:

```bash
bash tests/run_multiturn_maintenance.sh
```

That command runs the core Harbor unit tests for the overlay and the black-box three-round task regression manifest. See `TESTING.md` for the exact coverage.
