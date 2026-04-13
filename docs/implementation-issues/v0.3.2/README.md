# v0.3.2 Issues

This directory turns roadmap `v0.3.2` into execution-ready issues.

Status: `0.3.2-01` through `0.3.2-04` are complete on the current branch.

## Order

1. `0.3.2-01` SpireMeta aggregate importer and snapshot lineage
2. `0.3.2-02` STS2Runs public run archive contract and incremental sync
3. `0.3.2-03` Public run normalization and derived strategic stats
4. `0.3.2-04` Public-source consumption in priors, predictors, and benchmark reporting

## Critical Path

- `0.3.2-01` lands first because aggregate community imports should keep feeding the existing `community_card_stats` family instead of being redefined by later public-run work.
- `0.3.2-02` lands next because every later public-run feature depends on a stable raw archive contract, resumable sync rules, and dedupe behavior.
- `0.3.2-03` converts the raw archive into typed strategic summaries that downstream code can consume without scraping raw website payloads.
- `0.3.2-04` closes the loop by wiring those artifacts into priors, predictors, and benchmark/reporting outputs.

## Execution Waves

### Wave 1

- `0.3.2-01` SpireMeta aggregate importer and snapshot lineage
- `0.3.2-02` STS2Runs public run archive contract and incremental sync

These issues establish the two external source families and make incremental collection explicit instead of ad hoc.

Status:

- `0.3.2-01` complete on the current branch
- `0.3.2-02` complete on the current branch

### Wave 2

- `0.3.2-03` Public run normalization and derived strategic stats

This issue turns archived public run payloads into repo-native normalized summaries.

Status:

- `0.3.2-03` complete on the current branch

### Wave 3

- `0.3.2-04` Public-source consumption in priors, predictors, and benchmark reporting

This issue makes external public data operational inside the existing experiment stack.

## Guardrails

- aggregate card snapshots and public run archives remain separate artifact families
- public runs are analytical and prior-building inputs, not the primary BC / RL supervision dataset
- incremental sync must persist source lineage, cursors, dedupe metadata, and freshness diagnostics
- live training and evaluation must continue to work without any live website dependency

## Issues

- [0.3.2-01 SpireMeta Aggregate Importer And Snapshot Lineage](./0.3.2-01-spiremeta-aggregate-importer-and-snapshot-lineage.md)
- [0.3.2-02 STS2Runs Public Run Archive Contract And Incremental Sync](./0.3.2-02-sts2runs-public-run-archive-contract-and-incremental-sync.md)
- [0.3.2-03 Public Run Normalization And Derived Strategic Stats](./0.3.2-03-public-run-normalization-and-derived-strategic-stats.md)
- [0.3.2-04 Public-Source Consumption In Priors, Predictors, And Benchmark Reporting](./0.3.2-04-public-source-consumption-in-priors-predictors-and-benchmark-reporting.md)
