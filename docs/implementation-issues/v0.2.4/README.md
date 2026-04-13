# v0.2.4 Issues

This directory turns roadmap `v0.2.4` into execution-ready issues.

Status: `0.2.4-01` through `0.2.4-05` are complete on the current branch.

## Order

1. `0.2.4-01` Behavior cloning and imitation-learning pipeline: complete on current branch
2. `0.2.4-02` Offline RL dataset contract and trainer stack: complete on current branch
3. `0.2.4-03` Predictor calibration and ranking reports: complete on current branch
4. `0.2.4-04` Experiment registry and leaderboard outputs: complete on current branch
5. `0.2.4-05` Multi-job orchestration and experiment DAGs: complete on current branch

## Critical Path

- `0.2.4-01` establishes the first full supervised-training path and the checkpoint/report contract for learned policies.
- `0.2.4-02` extends the data and training stack into offline RL without inventing a second artifact shape.
- `0.2.4-03` makes predictor outputs measurable enough to influence promotion and rollback decisions.
- `0.2.4-04` turns raw artifacts into named experiments with lineage, aliases, and comparable summaries.
- `0.2.4-05` composes the previous outputs into resumable end-to-end experiment DAGs.

## Execution Waves

### Wave 1

- `0.2.4-01` Behavior cloning and imitation-learning pipeline
- `0.2.4-03` Predictor calibration and ranking reports

These two issues turn existing datasets and predictor exports into promotion-grade training and scoring products.

### Wave 2

- `0.2.4-02` Offline RL dataset contract and trainer stack
- `0.2.4-04` Experiment registry and leaderboard outputs

These two issues build on the Wave 1 artifact contracts and make cross-run comparison operational instead of ad hoc.

### Wave 3

- `0.2.4-05` Multi-job orchestration and experiment DAGs

This issue turns the earlier pieces into unattended multi-stage experiment flows.

## Issues

- [0.2.4-01 Behavior Cloning And Imitation-Learning Pipeline](./0.2.4-01-behavior-cloning-and-imitation-learning-pipeline.md)
- [0.2.4-02 Offline RL Dataset Contract And Trainer Stack](./0.2.4-02-offline-rl-dataset-contract-and-trainer-stack.md)
- [0.2.4-03 Predictor Calibration And Ranking Reports](./0.2.4-03-predictor-calibration-and-ranking-reports.md)
- [0.2.4-04 Experiment Registry And Leaderboard Outputs](./0.2.4-04-experiment-registry-and-leaderboard-outputs.md)
- [0.2.4-05 Multi-Job Orchestration And Experiment DAGs](./0.2.4-05-multi-job-orchestration-and-experiment-dags.md)
