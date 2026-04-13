# v0.3.4 Issues

This directory turns roadmap `v0.3.4` into execution-ready issues.

Status:

- `0.3.4-01` complete on the current branch
- `0.3.4-02` complete on the current branch
- `0.3.4-03` complete on the current branch
- `0.3.4-04` complete on the current branch
- `0.3.4-05` complete on the current branch

## Order

1. `0.3.4-01` Runtime semantic contract and build-lineage hardening
2. `0.3.4-02` Transactional selection planner and execution
3. `0.3.4-03` Strategic selection-fidelity datasets, training, and evaluation
4. `0.3.4-04` Quest, special-rest, and non-combat action-space completion
5. `0.3.4-05` Capability diagnostics and non-combat regression gates

## Critical Path

- `0.3.4-01` lands first because the repo should stop inferring core selection semantics and build identity from prompt text or weak runtime history whenever `STS2-Agent` can expose them directly.
- `0.3.4-02` lands next because multi-card selection is the most visible current capability gap and needs a transactional execution model before later data and benchmark work can trust these flows.
- `0.3.4-03` follows because stronger strategic datasets and fine-tuning coverage depend on the new runtime semantics plus the new transactional planner.
- `0.3.4-04` expands the same contract discipline into quest, special-rest, and other under-specified non-combat screens that still risk unsupported descriptors or ambiguous control semantics.
- `0.3.4-05` lands last because the benchmark and promotion gates should measure the completed contract and planner changes instead of locking in today's blind spots.

## Execution Waves

### Wave 1

- `0.3.4-01` Runtime semantic contract and build-lineage hardening

This issue establishes the runtime-side contract and ownership split the rest of the patch line depends on.

Status:

- `0.3.4-01` complete on the current branch
- `0.3.4-02` complete on the current branch
- `0.3.4-03` complete on the current branch
- `0.3.4-04` complete on the current branch
- `0.3.4-05` complete on the current branch

### Wave 2

- `0.3.4-02` Transactional selection planner and execution
- `0.3.4-03` Strategic selection-fidelity datasets, training, and evaluation

These issues convert the new runtime semantics into actual control behavior and training artifacts.

### Wave 3

- `0.3.4-04` Quest, special-rest, and non-combat action-space completion
- `0.3.4-05` Capability diagnostics and non-combat regression gates

These issues close remaining non-combat coverage gaps and make regressions measurable.

## Guardrails

- ownership must stay explicit: runtime state and action semantics belong to `STS2-Agent`; decision policy, datasets, training, eval, and regression gates belong to this repo
- ambiguous non-combat semantics must fail loudly or stay explicitly unsupported instead of being silently guessed from prompt text
- public strategic data remains initialization and analysis evidence, not control-equivalent runtime supervision
- build, beta-branch, and protocol drift must survive into datasets, checkpoints, and benchmark summaries
- new non-combat support is only considered complete when both execution and diagnostics are covered

## Issues

- [0.3.4-01 Runtime Semantic Contract And Build-Lineage Hardening](./0.3.4-01-runtime-semantic-contract-and-build-lineage-hardening.md)
- [0.3.4-02 Transactional Selection Planner And Execution](./0.3.4-02-transactional-selection-planner-and-execution.md)
- [0.3.4-03 Strategic Selection-Fidelity Datasets, Training, And Evaluation](./0.3.4-03-strategic-selection-fidelity-datasets-training-and-evaluation.md)
- [0.3.4-04 Quest, Special-Rest, And Non-Combat Action-Space Completion](./0.3.4-04-quest-special-rest-and-non-combat-action-space-completion.md)
- [0.3.4-05 Capability Diagnostics And Non-Combat Regression Gates](./0.3.4-05-capability-diagnostics-and-non-combat-regression-gates.md)
