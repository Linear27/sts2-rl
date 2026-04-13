# v0.3.0 Issues

This directory turns roadmap `v0.3.0` into execution-ready issues.

Status: `0.3.0-01` through `0.3.0-03` are complete on the current branch.

## Order

1. `0.3.0-01` Shadow combat encounter snapshots and dataset contract
2. `0.3.0-02` Shadow rollout and planner-search harness
3. `0.3.0-03` Search-guided evaluation and benchmark integration

## Critical Path

- `0.3.0-01` lands first because a shadow environment needs a stable encounter-start contract before any local planner or simulator work can be trusted.
- `0.3.0-02` turns those encounter snapshots into an executable research loop instead of a passive export.
- `0.3.0-03` closes the loop by comparing shadow-search guidance against live-runtime and benchmark artifacts.

## Execution Waves

### Wave 1

- `0.3.0-01` Shadow combat encounter snapshots and dataset contract

This issue creates the reusable encounter-start artifact layer needed for simulator or shadow-environment work.

Status: complete on the current branch.

### Wave 2

- `0.3.0-02` Shadow rollout and planner-search harness

This issue adds the executable local-research loop over selected combat encounters.

Status: complete on the current branch.

### Wave 3

- `0.3.0-03` Search-guided evaluation and benchmark integration

This issue pushes shadow-search results back into the repo's live evaluation, benchmark, and promotion flows.

Status: complete on the current branch.

## Issues

- [0.3.0-01 Shadow Combat Encounter Snapshots And Dataset Contract](./0.3.0-01-shadow-combat-encounter-snapshots-and-dataset-contract.md)
- [0.3.0-02 Shadow Rollout And Planner-Search Harness](./0.3.0-02-shadow-rollout-and-planner-search-harness.md)
- [0.3.0-03 Search-Guided Evaluation And Benchmark Integration](./0.3.0-03-search-guided-evaluation-and-benchmark-integration.md)
