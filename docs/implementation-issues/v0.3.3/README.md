# v0.3.3 Issues

This directory turns the next public-data training step into execution-ready issues.

Status:

- `0.3.3-01` complete on the current branch
- `0.3.3-02` complete on the current branch
- `0.3.3-03` complete on the current branch

## Order

1. `0.3.3-01` Public strategic decision dataset contract
2. `0.3.3-02` Strategic pretraining and public-decision trainer stack
3. `0.3.3-03` Mixed-data fine-tuning and evaluation integration

## Critical Path

- `0.3.3-01` lands first because public-run-derived training data needs its own dataset family instead of being forced into `trajectory_steps` or `offline_rl_transitions`.
- `0.3.3-02` lands next because strategic pretraining should consume only the supported non-combat decision records with explicit version, provenance, and reconstruction-confidence controls.
- `0.3.3-03` lands last because warm-start, mixed-data fine-tuning, and benchmark comparison only make sense after the dataset and pretraining contracts are stable.

## Guardrails

- public-run-derived training data must remain separate from `trajectory_steps` and `offline_rl_transitions`
- only supported strategic decision types may enter the public training corpus
- build / patch / beta-branch filters must be explicit and default to strict matching
- every training artifact must remain reproducible from local snapshots without live web calls
- source attribution, freshness, and downstream usage restrictions must survive into manifests and summaries

## Issues

- [0.3.3-01 Public Strategic Decision Dataset Contract](./0.3.3-01-public-strategic-decision-dataset-contract.md)
- [0.3.3-02 Strategic Pretraining And Public-Decision Trainer Stack](./0.3.3-02-strategic-pretraining-and-public-decision-trainer-stack.md)
- [0.3.3-03 Mixed-Data Fine-Tuning And Evaluation Integration](./0.3.3-03-mixed-data-fine-tuning-and-evaluation-integration.md)
