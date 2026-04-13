# v0.2.5 Issues

This directory turns roadmap `v0.2.5` into execution-ready issues.

Status: `0.2.5-01` through `0.2.5-04` are complete on the current branch.

## Order

1. `0.2.5-01` Runtime strategic-state contract for boss and map planning
2. `0.2.5-02` Boss-conditioned route planner and non-combat policy upgrade
3. `0.2.5-03` Boss/path-aware datasets, predictors, and benchmark suites
4. `0.2.5-04` Fixed-seed strategic evaluation and promotion gates

## Critical Path

- `0.2.5-01` must land first because the current runtime contract does not expose enough information for path planning.
- `0.2.5-02` turns the new state into an actually stronger controller instead of a passive logging upgrade.
- `0.2.5-03` makes the new strategic signals visible to BC, offline RL, and predictor flows so data quality improves instead of drifting.
- `0.2.5-04` closes the loop with fixed-seed suites and promotion rules that can tell whether strategic planning actually helped.

## Execution Waves

### Wave 1

- `0.2.5-01` Runtime strategic-state contract for boss and map planning
- `0.2.5-02` Boss-conditioned route planner and non-combat policy upgrade

These two issues establish the runtime information and direct control logic needed to stop making route choices from single-node heuristics alone.

### Wave 2

- `0.2.5-03` Boss/path-aware datasets, predictors, and benchmark suites

This issue upgrades the data products so the new strategic context reaches training and offline analysis instead of only the live heuristic path.

Status: complete on the current branch.

### Wave 3

- `0.2.5-04` Fixed-seed strategic evaluation and promotion gates

This issue turns the strategic planner into a measurable product with seed-stable regression gates.

## Issues

- [0.2.5-01 Runtime Strategic-State Contract For Boss And Map Planning](./0.2.5-01-runtime-strategic-state-contract-for-boss-and-map-planning.md)
- [0.2.5-02 Boss-Conditioned Route Planner And Non-Combat Policy Upgrade](./0.2.5-02-boss-conditioned-route-planner-and-non-combat-policy-upgrade.md)
- [0.2.5-03 Boss-Path-Aware Datasets Predictors And Benchmark Suites](./0.2.5-03-boss-path-aware-datasets-predictors-and-benchmark-suites.md)
- [0.2.5-04 Fixed-Seed Strategic Evaluation And Promotion Gates](./0.2.5-04-fixed-seed-strategic-evaluation-and-promotion-gates.md)
