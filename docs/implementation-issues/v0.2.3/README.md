# v0.2.3 Issues

This directory turns roadmap `v0.2.3` into execution-ready issues.

Status: complete on the current branch. `0.2.3-01` through `0.2.3-06` are implemented and covered by the current test suite.

## Order

1. `0.2.3-01` Dataset registry and reproducible builds
2. `0.2.3-02` Evaluation benchmark suites and statistical reporting
3. `0.2.3-03` Policy pack and combat hand planner
4. `0.2.3-04` Predictor integration into control and scoring
5. `0.2.3-05` Multi-instance job runner and watchdog
6. `0.2.3-06` Reproducibility and divergence diagnostics

## Critical Path

- `0.2.3-01` and `0.2.3-02` establish the data and evaluation contract for the rest of the patch line.
- `0.2.3-03` raises the non-learning baseline before more model complexity is introduced.
- `0.2.3-04` consumes the cleaned dataset and predictor path already in the repo.
- `0.2.3-05` turns the system from a single-run toolbox into an unattended multi-instance platform.
- `0.2.3-06` makes replay and comparison failures explainable enough for long-running experiments.

## Execution Waves

### Wave 1

- `0.2.3-01` Dataset registry and reproducible builds
- `0.2.3-02` Evaluation benchmark suites and statistical reporting

These two issues define the artifact contract that the rest of the patch line depends on.

### Wave 2

- `0.2.3-03` Policy pack and combat hand planner
- `0.2.3-04` Predictor integration into control and scoring

These two issues raise the repo's decision quality once data and evaluation are stable enough to measure the improvement.

### Wave 3

- `0.2.3-05` Multi-instance job runner and watchdog
- `0.2.3-06` Reproducibility and divergence diagnostics

These two issues turn the repo into a platform for unattended runs and make failures explainable.

## Issues

- [0.2.3-01 Dataset Registry And Reproducible Builds](./0.2.3-01-dataset-registry-and-reproducible-builds.md)
- [0.2.3-02 Evaluation Benchmark Suites And Statistical Reporting](./0.2.3-02-evaluation-benchmark-suites-and-statistical-reporting.md)
- [0.2.3-03 Policy Pack And Combat Hand Planner](./0.2.3-03-policy-pack-and-combat-hand-planner.md)
- [0.2.3-04 Predictor Integration Into Control And Scoring](./0.2.3-04-predictor-integration-into-control-and-scoring.md)
- [0.2.3-05 Multi-Instance Job Runner And Watchdog](./0.2.3-05-multi-instance-job-runner-and-watchdog.md)
- [0.2.3-06 Reproducibility And Divergence Diagnostics](./0.2.3-06-reproducibility-and-divergence-diagnostics.md)
