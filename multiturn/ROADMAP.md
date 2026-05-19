# Multiturn Roadmap

This roadmap tracks the local multiturn Harbor overlay. It is scoped to
`multiturn/main` and keeps the current public contract stable while reducing
future upstream merge cost.

## Compatibility Contract

All roadmap items are additive or internal unless a future document explicitly
marks a breaking change. The following interfaces stay compatible:

- Existing `harbor run` CLI flags and current flag combinations.
- Existing task layout: top-level `task.toml`, `round_N/instruction.md`,
  `round_N/solution/solve.*`, and `round_N/tests/test.*`.
- Existing persisted config fields under `verifier.multiround_*`.
- Existing output paths under `verifier/`, `state/`, and `agent/`.
- Existing black-box test entrypoints in the parent `tests/` directory.
- Existing maintenance branch model: merge upstream `origin/main` into
  `multiturn/main` with merge commits on the maintained integration branch.

New behavior should be enabled by new flags, new optional metadata, or internal
helpers with unchanged user-visible defaults.

## P0: Lower Upstream Merge Friction

Goal: keep multiturn behavior stable while shrinking the patch surface inside
large upstream Harbor files.

- Move pure resume output planning into `harbor.multiround.resume_plan`.
- Keep `cli/jobs.py` responsible for existing CLI parsing, filesystem mutation,
  backup, cleanup, and config persistence.
- Add direct unit tests for resume planning so upstream CLI churn can be
  audited separately from multiturn semantics.
- Keep multiturn docs and CI in the local overlay and parent harness, separate
  from Harbor upstream docs/workflows.

Current status:

- Done: resume output planning lives in `harbor.multiround.resume_plan` with
  direct unit coverage.
- Next: apply the same pattern to roundwise fanout selection once upstream
  merge pressure or feature work touches that area.

## P1: Safer Resume Operations

Goal: make resume intent easier to inspect before mutating a source trial.

- Add a dry-run resume report that prints the resolved source trial, inferred
  jobs directory, backup behavior, resume round, preflight policy, and snapshot
  source.
- Add an optional explicit resume mode flag while preserving today’s default:
  in-place with automatic backup.
- Keep `--output-jobs-dir` as the non-mutating copy mode and
  `--no-resume-backup` as the expert in-place mode.
- Extend black-box coverage when a new visible flag is added.

Current status:

- Done: `--resume-dry-run` reports the resolved source, output mode, jobs
  directory, backup intent, planned resume source, and config trial name as an
  inspection-only command. It is covered by direct unit tests and the black-box
  parameter validation case.
- Next: add an explicit resume mode flag only if operators need a clearer
  spelling than the existing `--output-jobs-dir` / `--no-resume-backup` split.

## P2: Fanout Selection and Lineage Observability

Goal: make MT@K selection behavior easier to audit while preserving default
ranking.

- Isolate frontier selection into a pure planner module.
- Keep current first-success parent selection as the default.
- Add optional future selection policies only behind explicit flags.
- Persist a compact lineage summary for each child trial and expose it to the
  viewer while preserving existing artifact paths.

## P3: Artifact Schema and Viewer Support

Goal: make generated task regressions and human debugging easier.

- Add schema/golden tests for `multiround_results.json`, snapshot metadata,
  resume lineage metadata, and trajectory round annotations.
- Build a viewer-side multiturn summary from existing artifacts first.
- Add optional richer viewer metadata after the summary is stable.

Necessity:

- Done: `tests/validate_multiturn_artifacts.py` validates the current stable
  trial artifact shape for `multiround_results.json`, snapshot metadata, and
  resume lineage metadata. The Oracle full-run smoke case invokes it for the
  3-round baseline.
- Schema/golden checks are required for machine-consumed artifacts that feed
  CI, evaluator summaries, resume preflight, and future viewer pages.
- The first useful target is the stable artifact set:
  `multiround_results.json`, `state/round_N/snapshot.json`,
  `agent/pre_resume_*/metadata.json`, and trajectory round annotations.

## P4: Environment Snapshot Portability

Goal: support more Harbor environments while preserving Docker behavior.

- Keep Docker snapshot capture/restore as the reference implementation.
- Add capability checks and clearer diagnostics for non-Docker environments.
- Add remote-environment snapshot metadata only when an environment advertises
  the capability.

Resume ownership model:

- Harbor owns resume semantics: round selection, source trial resolution,
  snapshot metadata selection, artifact copy/cleanup, lineage recording, and
  aggregate reward windows.
- The environment adapter owns state materialization: creating a runnable
  workspace/container from a snapshot, capturing a new snapshot at the end of a
  round, and returning portable metadata to Harbor.
- This split keeps trial artifacts, CLI behavior, and future viewer summaries
  stable across local Docker and remote providers.

Current Daytona position:

- Harbor's Docker environment implements the current per-round state snapshot
  contract through `capture_state_snapshot()`.
- Daytona supports remote sandboxes, Docker/DinD execution, and Daytona-level
  startup snapshots.
- Daytona-backed fresh multiturn runs use the normal Harbor remote environment
  path.
- Daytona-backed resume / roundwise becomes a supported path when the Daytona
  adapter advertises Harbor-compatible round snapshot capture/restore metadata.
- The current maintenance gate includes local Daytona preflight; the next
  capability step is a credentialed smoke gate that records the adapter
  snapshot fields Harbor needs.

## P5: Maintenance Automation

Goal: catch upstream Harbor drift earlier.

- Keep the deterministic parent workflow as the required merge gate.
- Keep upstream-merge rehearsal as a documented manual detached-worktree command.
- Keep credentialed Terminus-2 MT@4 and SR tests in the release-like validation
  tier where operators explicitly provide local credentials and runtime budget.
- Record stable commands and current expected outputs in docs. Runtime logs and
  credentials stay in local ignored paths.

Automation boundary is documented in `MAINTENANCE_AUTOMATION.md`.
