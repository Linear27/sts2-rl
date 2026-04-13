# v0.3.1 Issues

This directory turns roadmap `v0.3.1` into execution-ready issues.

Status: `0.3.1-01` through `0.3.1-03` are complete on the current branch.

## Order

1. `0.3.1-01` Community card stats snapshot contract
2. `0.3.1-02` Reward and shop prior integration
3. `0.3.1-03` Community alignment benchmark metrics

## Critical Path

- `0.3.1-01` lands first because reward/shop priors need a stable imported snapshot contract before policy logic can consume community signals safely.
- `0.3.1-02` turns those imported snapshots into decision-time priors for card reward and shop actions.
- `0.3.1-03` closes the loop by evaluating whether repo policies align with high-signal community card decisions.

## Execution Waves

### Wave 1

- `0.3.1-01` Community card stats snapshot contract

This issue creates the local artifact layer for importing community pick-rate and win-delta snapshots.

Status: complete on the current branch.

### Wave 2

- `0.3.1-02` Reward and shop prior integration

This issue injects imported community priors into reward and shop scoring without turning them into hard labels.

Status: complete on the current branch.

### Wave 3

- `0.3.1-03` Community alignment benchmark metrics

This issue adds alignment metrics and reporting over imported community card signals.

Status: complete on the current branch.

## Issues

- [0.3.1-01 Community Card Stats Snapshot Contract](./0.3.1-01-community-card-stats-snapshot-contract.md)
- [0.3.1-02 Reward And Shop Prior Integration](./0.3.1-02-reward-and-shop-prior-integration.md)
- [0.3.1-03 Community Alignment Benchmark Metrics](./0.3.1-03-community-alignment-benchmark-metrics.md)
