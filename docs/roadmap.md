# Roadmap

## v0.3.5

Status:

- `0.3.5-01` complete on the current branch

Issue breakdown:

- [0.3.5 issue index](./implementation-issues/v0.3.5/README.md)
- [0.3.5-01 Custom-run action-space and contract-native configuration](./implementation-issues/v0.3.5/0.3.5-01-custom-run-action-space-and-contract-native-configuration.md)

### Goal

Close the remaining repo-side `CUSTOM_RUN` ownership gap by consuming the runtime's full custom-run configuration contract directly instead of leaving parameterized descriptors unsupported or routing preparation through per-modifier toggles.

### Why This Is The Right Next Step

The current branch already has:

- deterministic runtime normalization and fixed-seed custom-run preparation
- explicit repo-vs-`STS2-Agent` ownership language for non-combat capability gaps
- capability diagnostics that treat unsupported descriptors as release-significant regressions

The next bottleneck is a narrow but real contract hole:

- `STS2-Agent` already exposes `set_custom_seed`, `set_custom_ascension`, and `set_custom_modifiers`
- this repo still treated those parameterized descriptors as second-class and left them out of `candidate_actions`
- custom-run preparation still configured modifiers through incremental `toggle_custom_modifier` calls instead of the runtime's atomic bundle-set action
- that mismatch could create false repo-side capability warnings on `CUSTOM_RUN` and keep the repo from consuming the authoritative runtime contract cleanly

That direction stays inside the intended boundary:

- `STS2-Agent` owns the authoritative custom-run state, supported actions, and parameter semantics
- this repo owns action-space materialization, contract-aware preparation flows, tests, and diagnostics about unsupported descriptors

### Scope

1. Custom-run action-space completion

- materialize runtime-backed candidate actions for `set_custom_seed`, `set_custom_ascension`, and `set_custom_modifiers`
- keep parameterized custom-run descriptors out of the unsupported-action bucket

2. Contract-native preparation

- switch custom-run preparation from incremental modifier toggles to atomic `set_custom_modifiers`
- validate requested seed, ascension, and modifier bundles against the runtime contract before embark

3. Regression coverage

- extend custom-run tests to cover the completed action space and atomic modifier configuration
- keep deterministic action-sequence assertions for custom-run setup

### Delivered On The Current Branch

- `build_candidate_actions` now expands `CUSTOM_RUN` descriptors for `set_custom_seed`, bounded `set_custom_ascension`, and current-bundle `set_custom_modifiers`
- custom-run preparation now requires the runtime configuration descriptors it depends on instead of silently assuming partial support
- modifier configuration now applies through a single `set_custom_modifiers` request and verifies the resulting selected modifier bundle before embark
- regression tests now cover the expanded `CUSTOM_RUN` action space, empty-bundle clearing, and multi-modifier atomic bundle application

### Exit Criteria

`v0.3.5` is complete when the repo can do all of the following:

- consume the runtime's `CUSTOM_RUN` configuration descriptors without surfacing repo-side unsupported warnings for those actions
- configure requested seed, ascension, and modifier bundles through the contract-native runtime actions
- verify custom-run preparation deterministically through targeted regression tests and action-sequence assertions

## v0.3.4

Status:

- `0.3.4-01` complete on the current branch
- `0.3.4-02` complete on the current branch
- `0.3.4-03` complete on the current branch
- `0.3.4-04` complete on the current branch
- `0.3.4-05` complete on the current branch

Issue breakdown:

- [0.3.4 issue index](./implementation-issues/v0.3.4/README.md)
- [0.3.4-01 Runtime semantic contract and build-lineage hardening](./implementation-issues/v0.3.4/0.3.4-01-runtime-semantic-contract-and-build-lineage-hardening.md)
- [0.3.4-02 Transactional selection planner and execution](./implementation-issues/v0.3.4/0.3.4-02-transactional-selection-planner-and-execution.md)
- [0.3.4-03 Strategic selection-fidelity datasets, training, and evaluation](./implementation-issues/v0.3.4/0.3.4-03-strategic-selection-fidelity-datasets-training-and-evaluation.md)
- [0.3.4-04 Quest, special-rest, and non-combat action-space completion](./implementation-issues/v0.3.4/0.3.4-04-quest-special-rest-and-non-combat-action-space-completion.md)
- [0.3.4-05 Capability diagnostics and non-combat regression gates](./implementation-issues/v0.3.4/0.3.4-05-capability-diagnostics-and-non-combat-regression-gates.md)

### Goal

Turn the current strategic control stack into a capability-bounded non-combat planner with explicit runtime semantics, transactional deck-selection execution, and build-tight data consumption instead of relying on prompt inference and partial special cases.

### Why This Is The Right Next Step

The current branch already has:

- policy-pack coverage for route, reward, shop, selection, event, and rest decisions
- community prior integration for reward, shop, and selection scoring
- public-run strategic datasets plus strategic pretraining and runtime-side fine-tuning
- live strategic checkpoint integration inside collect, train, and evaluation flows

The next bottleneck is no longer basic non-combat coverage. It is fidelity and ownership:

- multi-card selection flows are still treated as repeated single-card clicks rather than one transactional decision
- selection semantics are still inferred from prompt text and narrow history checks instead of being carried explicitly by the runtime contract
- unsupported action descriptors and `policy_no_action_timeout` stops still exist, but the current artifacts do not cleanly attribute whether the gap belongs to this repo or `STS2-Agent`
- Early Access patch churn and beta-value drift make build- and branch-tight data consumption mandatory instead of optional metadata hygiene
- quest, special rest-site, and other follow-up non-combat screens stretch the current action contract beyond the already-supported shop-remove special case

That direction matches the repo split:

- this repo owns decision logic, training, dataset contracts, evaluation, and regression gates
- `STS2-Agent` owns the authoritative runtime state and action contract exposed through the local HTTP API
- the next patch line should therefore make cross-repo ownership explicit instead of burying contract gaps inside policy heuristics

### Planned Themes

- explicit repo-vs-`STS2-Agent` ownership for non-combat capability gaps
- runtime semantic contracts for selection source, mode, transaction boundaries, and build lineage
- transactional multi-card selection, remove, upgrade, pick, and transform execution in the repo policy layer
- strategic dataset, trainer, and benchmark expansion that respects stronger selection semantics and strict build matching
- measurable diagnostics and release gates for unsupported descriptors, no-action stalls, and non-combat coverage drift

### Explicit Non-Goals For v0.3.4

- using prompt parsing as the long-term primary source of runtime selection semantics
- treating public strategic data as control-equivalent `trajectory_steps` or offline-RL transitions
- shipping broad multiplayer or co-op strategy logic beyond the contract and control-scope coverage needed to avoid silent failures
- claiming support for new non-combat screens without explicit runtime tests, diagnostics, and artifact lineage

### Delivered On The Current Branch

- `STS2-Agent` now exposes explicit runtime build lineage in both `/health` and `/state`, including `build_id`, `branch`, `content_channel`, `commit`, `build_date`, and `main_assembly_hash`
- `/state.selection` now carries explicit `selection_family`, `semantic_mode`, `source_type`, `source_room_type`, prompt localization keys, required and remaining counts, and multi-select capability flags
- deck-selection transaction metadata now uses runtime-selected counts and confirmation availability for non-combat selection screens instead of only the combat-hand path
- this repo now ingests the expanded runtime contract through typed payload models, state summaries, and strategic runtime context payloads
- selection policy and strategic extraction paths now consume explicit runtime semantics and source types instead of treating prompt parsing and previous-screen history as the primary control signal
- regression coverage now includes expanded health parsing, state build parsing, explicit-semantic selection policy behavior, and runtime strategic extraction under the new contract
- selection control now executes deck pick, remove, upgrade, and transform flows through a persistent policy-side transaction planner instead of stateless per-frame card clicks
- transaction execution now reconciles runtime selected counts, replans remaining bundle members after intermediate divergence, and confirms only after the planned requirement is satisfied
- decision traces now persist `selection_transaction` metadata with transaction id, phase, planned members, completed members, next member, and recovery diagnostics
- trajectory state summaries now preserve selection confirmation fields alongside explicit selection counts so downstream artifacts can distinguish partial vs confirmable bundle state
- strategic decision taxonomy now includes generalized runtime selection families such as `selection_remove`, `selection_upgrade`, and `selection_transform` instead of a narrow `shop_remove` special case
- runtime fine-tuning now extracts explicit selection decisions from runtime deck-selection screens using transaction metadata, source semantics, and build lineage instead of prior shop-trigger reconstruction
- public strategic datasets now materialize generalized `selection_remove` records with explicit source typing, plus richer build-lineage fields and summary histograms for game version, branch, and content channel when available
- strategic runtime guidance now consumes the expanded selection decision families directly on the live selection hook, including non-shop remove and upgrade flows
- strategic fine-tune summaries and checkpoints now preserve runtime/public dataset decision histograms plus build-lineage histograms, and mixed runtime/public training now rejects disjoint build lineage by default
- event and rest follow-up screens now carry explicit runtime lineage into `selection` and `reward`, including source action, event ids and option keys, and rest option ids instead of collapsing back to generic room-only context
- reward-screen strategic routing now honors explicit runtime source typing for event-linked follow-up rewards instead of defaulting them to monster rewards
- rest-site policy scoring now treats canonical `option_id` values as first-class semantics, so special or relabeled rest options do not depend on fragile UI-title matching

### Exit Criteria

`v0.3.4` is complete when the repo can do all of the following:

- consume explicit runtime metadata for build identity and targeted non-combat semantics, or fail loudly when the runtime contract is too ambiguous
- execute multi-card selection bundles as one transactional policy unit with confirmation, recovery, and trace metadata
- train and evaluate strategic selection models with build-tight lineage and stronger selection decision typing than the current shop-remove-only bridge
- attribute unsupported non-combat failures to repo-side policy gaps versus `STS2-Agent` contract gaps through persisted diagnostics
- document the ownership boundary and execution order clearly enough that follow-up implementation work can land without reopening scope confusion

## v0.2.3

Status: complete on the current branch. `0.2.3-01` through `0.2.3-06` are complete on the current branch.

Issue breakdown:

- [0.2.3 issue index](./implementation-issues/v0.2.3/README.md)
- [0.2.3-01 Dataset registry and reproducible builds](./implementation-issues/v0.2.3/0.2.3-01-dataset-registry-and-reproducible-builds.md)
- [0.2.3-02 Evaluation benchmark suites and statistical reporting](./implementation-issues/v0.2.3/0.2.3-02-evaluation-benchmark-suites-and-statistical-reporting.md)
- [0.2.3-03 Policy pack and combat hand planner](./implementation-issues/v0.2.3/0.2.3-03-policy-pack-and-combat-hand-planner.md)
- [0.2.3-04 Predictor integration into control and scoring](./implementation-issues/v0.2.3/0.2.3-04-predictor-integration-into-control-and-scoring.md)
- [0.2.3-05 Multi-instance job runner and watchdog](./implementation-issues/v0.2.3/0.2.3-05-multi-instance-job-runner-and-watchdog.md)
- [0.2.3-06 Reproducibility and divergence diagnostics](./implementation-issues/v0.2.3/0.2.3-06-reproducibility-and-divergence-diagnostics.md)

### Goal

Turn the current single-machine training workspace into a repeatable experiment platform with stronger data management, stronger baselines, and evaluation that is good enough to drive iteration without manual log digging.

### Why This Is The Right Next Step

The current branch already has:

- runtime normalization and anchored starts
- replay and checkpoint comparison artifacts
- a predictor bootstrap path
- a combat-only masked DQN
- schedule-side checkpoint promotion

The next bottleneck is no longer "can we run training?" but:

- can we build reliable datasets from artifacts
- can we compare model changes on a stable benchmark suite
- can we raise the heuristic and planner baseline before adding more learning complexity
- can we keep multiple real instances healthy long enough to generate consistent data

That direction is aligned with the practical lessons from `CommunicationMod`, `spirecomm`, `runlogger`, `bottled_ai`, `SlayTheSpireFightPredictor`, `SeedSearch`, `MiniSTS`, and `conquer-the-spire`.

### Scope

1. Dataset registry and reproducible builds

- add dataset manifests, split metadata, and schema versioning
- build train/validation/test datasets from trajectory and combat-outcome artifacts
- support dataset filtering, weighting, and export for predictor, BC, and offline-RL consumers

2. Evaluation benchmark suites and statistical reporting

- define benchmark suites by seed, floor band, combat type, and checkpoint pair
- add aggregate reports with confidence intervals or bootstrap summaries
- promote evaluation outputs from ad hoc summaries into experiment-grade reports

3. Policy pack and combat hand planner

- split heuristics into explicit policy packs instead of one monolithic simple policy
- add a bounded combat hand planner or search-based action ranker
- add policy regression tests and coverage reporting

4. Predictor integration

- use the existing predictor branch in route, reward, shop, and combat scoring paths
- persist predictor-side features and scores into artifacts
- expose predictor-guided eval and ablation modes

5. Multi-instance job runner and watchdog

- add a repo-native runner for collect, eval, and train jobs across several instances
- add health checks, retries, cooldowns, and quarantine behavior
- make multi-instance failure modes visible in structured logs

6. Reproducibility and divergence diagnostics

- pin start-state metadata and environment fingerprints
- explain replay/compare divergence beyond a single status string
- surface preparation drift, runtime drift, and policy drift separately

### Exit Criteria

`v0.2.3` is complete when the repo can do all of the following:

- build reproducible datasets with manifests from collected artifacts
- run a benchmark suite and compare checkpoints through stable statistical summaries
- run stronger heuristic or planner baselines with explicit regression coverage
- operate several instances under a single repo-native job runner with failure handling
- diagnose replay or comparison divergence with explicit structured categories

## v0.2.5

Status: `0.2.5-01` through `0.2.5-04` are complete on the current branch.

Issue breakdown:

- [0.2.5 issue index](./implementation-issues/v0.2.5/README.md)
- [0.2.5-01 Runtime strategic-state contract for boss and map planning](./implementation-issues/v0.2.5/0.2.5-01-runtime-strategic-state-contract-for-boss-and-map-planning.md)
- [0.2.5-02 Boss-conditioned route planner and non-combat policy upgrade](./implementation-issues/v0.2.5/0.2.5-02-boss-conditioned-route-planner-and-non-combat-policy-upgrade.md)
- [0.2.5-03 Boss-path-aware datasets, predictors, and benchmark suites](./implementation-issues/v0.2.5/0.2.5-03-boss-path-aware-datasets-predictors-and-benchmark-suites.md)
- [0.2.5-04 Fixed-seed strategic evaluation and promotion gates](./implementation-issues/v0.2.5/0.2.5-04-fixed-seed-strategic-evaluation-and-promotion-gates.md)

### Goal

Turn the current non-combat controller from a local node scorer into a strategic planner that can reason about the current boss, upcoming route structure, and act-level preparation under fixed-seed live evaluation.

### Why This Is The Right Next Step

The current branch already has:

- deterministic custom-run preparation for fixed-seed live evaluation
- policy-pack control with a stronger combat hand planner
- predictor-guided scoring and manifest-built datasets
- offline RL, behavior cloning, benchmark suites, and experiment registry flows

The next bottleneck is strategic blindness outside combat:

- current route choice is still immediate-node scoring rather than path-prefix planning
- boss identity and act-level demands are not first-class runtime inputs
- datasets and predictors still summarize map state too coarsely for strategic learning
- fixed-seed progress can stall because the same strategic mistakes repeat on the same map

That direction is aligned with practical lessons from `CommunicationMod`, `spirecomm`, `infomod2`, `Better Paths`, and route-planning tools such as `sts2routesuggest`: Slay the Spire agents need boss-aware, path-aware state to make non-combat decisions that are worth imitating or optimizing.

### Scope

1. Runtime strategic-state contract

- expose boss identity, act metadata, and a route-plannable map graph
- persist the new strategic fields through typed models and trajectory summaries
- keep compatibility with older artifacts where feasible

2. Boss-conditioned route planner

- replace single-node route scoring with bounded path-prefix planning
- propagate strategic context into rest, reward, and shop decisions
- persist ranked route alternatives and scores into decision traces

3. Strategic datasets and predictors

- add boss/path-aware fields to trajectory, predictor, and offline-RL datasets
- extend predictor feature maps and benchmark slices with strategic context
- keep deterministic manifests and summary versioning intact

4. Fixed-seed strategic evaluation

- add route-aware benchmark suites and promotion thresholds
- compare legacy and strategic planners on the same fixed seeds
- track route-quality metrics alongside run outcomes

### Exit Criteria

`v0.2.5` is complete when the repo can do all of the following:

- observe boss identity and a route-plannable map structure from the live runtime
- choose map actions through bounded path planning instead of only immediate-node scoring
- export boss/path-aware datasets and predictor features for later learning
- evaluate and promote strategic planners through fixed-seed route-aware benchmark suites

## v0.2.4

Status: `0.2.4-01` through `0.2.4-05` are complete on the current branch.

Issue breakdown:

- [0.2.4 issue index](./implementation-issues/v0.2.4/README.md)
- [0.2.4-01 Behavior cloning and imitation-learning pipeline](./implementation-issues/v0.2.4/0.2.4-01-behavior-cloning-and-imitation-learning-pipeline.md)
- [0.2.4-02 Offline RL dataset contract and trainer stack](./implementation-issues/v0.2.4/0.2.4-02-offline-rl-dataset-contract-and-trainer-stack.md)
- [0.2.4-03 Predictor calibration and ranking reports](./implementation-issues/v0.2.4/0.2.4-03-predictor-calibration-and-ranking-reports.md)
- [0.2.4-04 Experiment registry and leaderboard outputs](./implementation-issues/v0.2.4/0.2.4-04-experiment-registry-and-leaderboard-outputs.md)
- [0.2.4-05 Multi-job orchestration and experiment DAGs](./implementation-issues/v0.2.4/0.2.4-05-multi-job-orchestration-and-experiment-dags.md)

### Goal

Turn the experiment platform into a full training stack that can support imitation learning and offline RL without reworking the repo shape again.

### Why This Is The Right Next Step

The current branch now has:

- manifest-driven dataset builds with lineage and deterministic splits
- benchmark suites with statistical summaries
- policy packs, planner-enhanced heuristics, and predictor-guided scoring
- multi-instance runtime jobs with watchdogs
- replay and checkpoint comparison diagnostics with structured divergence causes

That means the next bottleneck is no longer data collection or runtime observability. It is the lack of a full training stack above those foundations:

- there is still no first-class behavior cloning trainer that consumes the repo's filtered datasets
- there is still no repo-native offline RL pipeline that can learn from static trajectory corpora before live fine-tuning
- predictor calibration, ranking, and benchmark comparison reports
- a local experiment registry with aliases, promotion history, leaderboard outputs, and compare reports
- schedules can chain repeated training sessions, but they cannot yet express multi-stage DAGs like collect -> build dataset -> train -> eval -> compare -> promote

That direction is aligned with practical patterns from `HumanCompatibleAI/imitation`, `d3rlpy`, `CORL`, `Minari`, `CommunicationMod`, `SlayTheSpireFightPredictor`, `DVC`, and `MLflow`.

### Planned Themes

- behavior cloning / imitation-learning pipeline from filtered datasets
- offline RL data contracts and trainers
- predictor calibration and ranking reports
- experiment registry and leaderboard outputs: complete on current branch
- richer schedule orchestration across multiple jobs

### Scope

1. Behavior cloning and imitation learning

- train supervised policies directly from the repo's trajectory datasets
- support filtered datasets, weighting, held-out evaluation, and checkpoint export
- add rollout evaluation against the live runtime and benchmark suites

2. Offline RL data contracts and trainers

- add transition- and episode-level offline RL dataset contracts on top of current trajectory exports
- support at least one practical discrete offline RL baseline before broadening the algorithm set
- preserve compatibility with later offline-to-online fine-tuning instead of building a dead-end pipeline

3. Predictor calibration and ranking

- evaluate predictor reliability with calibration metrics and plots
- evaluate ranking quality for reward, shop, and route-choice use cases
- turn predictor promotion from ad hoc inspection into measurable policy support

4. Experiment registry and leaderboard

- promote important artifacts into a repo-native experiment registry with aliases, lineage, and comparable metrics
- add leaderboard-style summaries across BC, offline RL, heuristic, and predictor-guided runs
- keep the registry local-first and deterministic, with optional external exporters later

5. Multi-job orchestration

- extend the current runtime job runner into a multi-stage DAG orchestrator
- support end-to-end experiment flows across collect, dataset build, train, eval, compare, and promote
- add resumability, stage-level retries, and explicit artifact handoff between jobs

### Reference-Driven Decisions

- `HumanCompatibleAI/imitation` supports the choice to treat BC as a first-class training product rather than a one-off notebook path.
- `d3rlpy` and `CORL` support prioritizing practical offline RL baselines with strong experiment structure before exploring more novel algorithms.
- `Minari` supports keeping dataset contracts explicit and reusable instead of relying on ad hoc replay-buffer serialization.
- `SlayTheSpireFightPredictor` supports dedicating a full patch issue to predictor calibration and ranking quality instead of treating predictor outputs as opaque scores.
- `DVC` and `MLflow` support building a registry-oriented experiment layer with versioned metrics, aliases, and lineage rather than leaving artifacts as unnamed folders.

### Exit Criteria

`v0.2.4` is complete when the repo can do all of the following:

- train and evaluate a behavior-cloned policy from manifest-built datasets
- materialize offline RL datasets and train at least one stable offline RL baseline with repo-native commands
- produce predictor calibration and ranking reports that are good enough to inform promotion or rollback
- register experiments, compare them on a leaderboard, and promote named artifacts with explicit lineage
- orchestrate a multi-stage experiment DAG that runs across collection, dataset build, training, evaluation, and promotion

## v0.3.1

Status: `0.3.1-01` through `0.3.1-03` are complete on the current branch.

Issue breakdown:

- [0.3.1 issue index](./implementation-issues/v0.3.1/README.md)
- [0.3.1-01 Community card stats snapshot contract](./implementation-issues/v0.3.1/0.3.1-01-community-card-stats-snapshot-contract.md)
- [0.3.1-02 Reward and shop prior integration](./implementation-issues/v0.3.1/0.3.1-02-reward-and-shop-prior-integration.md)
- [0.3.1-03 Community alignment benchmark metrics](./implementation-issues/v0.3.1/0.3.1-03-community-alignment-benchmark-metrics.md)

### Goal

Turn external community pick-rate / buy-rate / win-delta signals into repo-native priors that can later guide reward, shop, and benchmark analysis without depending on live web scraping.

### Planned Themes

- canonical local snapshots for community card statistics
- policy-pack reward / shop priors derived from imported community signals
- alignment metrics comparing repo decisions against community pick/buy distributions

### Delivered On The Current Branch

- canonical `community_card_stats` artifact contract with typed records, CSV/JSONL exports, and summary generation
- repo-native `sts2-rl community import` and `sts2-rl community summary` commands for importing local CSV/JSON/JSONL snapshots
- community snapshot summaries with source / character / source-type histograms plus pick-rate, buy-rate, and win-delta stats
- policy-pack community prior module with character / ascension / act / floor-band matching and confidence-scaled reward/shop/remove bonuses
- collect, live eval, checkpoint eval, runtime job, and benchmark suite config paths now accept explicit community prior runtime configs
- decision traces now surface selected and ranked community prior contributions so reward/shop audits do not require score reconstruction
- tests covering CSV import, embedded JSON metadata, rate derivation, and CLI summary flows
- benchmark eval and compare cases now emit community alignment summaries, per-domain breakdowns, and baseline-vs-candidate delta metrics from raw step logs
- suite summaries now aggregate eval and compare community alignment so imported priors can be tracked across experiments

## v0.3.3

Status:

- `0.3.3-01` complete on the current branch
- `0.3.3-02` complete on the current branch
- `0.3.3-03` complete on the current branch

Issue breakdown:

- [0.3.3 issue index](./implementation-issues/v0.3.3/README.md)
- [0.3.3-01 Public strategic decision dataset contract](./implementation-issues/v0.3.3/0.3.3-01-public-strategic-decision-dataset-contract.md)
- [0.3.3-02 Strategic pretraining and public-decision trainer stack](./implementation-issues/v0.3.3/0.3.3-02-strategic-pretraining-and-public-decision-trainer-stack.md)
- [0.3.3-03 Mixed-data fine-tuning and evaluation integration](./implementation-issues/v0.3.3/0.3.3-03-mixed-data-fine-tuning-and-evaluation-integration.md)

### Goal

Turn public runs into a first-class strategic pretraining corpus without confusing partial public supervision with step-level runtime control data.

### Why This Is The Right Next Step

The current branch already has:

- lineage-rich aggregate public snapshots
- incrementally synced public-run archives plus normalized strategic summaries
- public-source consumption in priors, predictors, and benchmark reports
- runtime-side BC, offline RL, benchmark, and registry infrastructure

The next bottleneck is no longer access to public data. It is how to use that data for training without violating supervision boundaries:

- public runs clearly contain high-value strategic signals for route, reward, shop, event, and rest decisions
- public runs still do not match the repo's runtime-faithful `trajectory_steps` or `offline_rl_transitions` contracts
- early-access and public-beta patch churn makes strict version and build filters mandatory
- large public corpora are most valuable as initialization and strategic supervision before higher-fidelity local fine-tuning

That direction is aligned with both the repo's current architecture and the practical shape of the public ecosystem:

- public APIs and trackers expose enough run and aggregate information to build large local strategic corpora
- source terms, cache behavior, and patch drift all favor local snapshot artifacts over live training dependencies
- passive-data and observation-only learning evidence supports strategic pretraining and warm-start usage more strongly than direct full-control imitation

### Planned Themes

- `public_strategic_decisions` as a dedicated dataset family
- strategic pretraining trainers and checkpoint contracts
- warm-start and mixed-data fine-tuning on top of runtime-collected control data
- provenance, build matching, and promotion guardrails for public-data-assisted experiments

### Delivered On The Current Branch

- `public_strategic_decisions` is now a first-class dataset kind and local artifact family
- dataset manifests can now consume `public_run_normalized` sources directly through the existing dataset builder
- reward and shop-buy examples preserve full candidate sets, while shop-remove, event-choice, and rest-site actions are tagged as explicit `chosen_only` weak supervision
- dataset summaries and CSV exports now surface build, source, decision-type, support-quality, and confidence diagnostics for public strategic decisions
- regression tests cover manifest resolution, filtering, deterministic dataset materialization, and source-aware weak-supervision extraction
- `strategic_pretrain` is now a first-class trainer with dedicated checkpoint and summary contracts
- strategic pretraining now learns from both strong full-candidate records and explicit positive-only weak-supervision records instead of collapsing them into one objective
- held-out summaries now report ranking, positive-only, auxiliary value, and per-decision-type diagnostics for public strategic datasets
- CLI training flows can now materialize strategic pretraining artifacts directly from `public_strategic_decisions` datasets
- `strategic_finetune` is now a first-class runtime-side trainer that extracts only explicit strategic decisions from local `trajectory_steps`
- fine-tuning can now warm-start from `strategic_pretrain` or prior `strategic_finetune` checkpoints with explicit feature-schema checks and transferred-module metadata
- mixed runtime/public schedules now keep sources separate in loaders, summaries, metrics, and registry lineage instead of flattening public supervision into runtime control data
- training summaries and registry inspection now preserve runtime lineage, public lineage, warm-start checkpoint references, and transferred-module freeze schedule metadata
- CLI and regression coverage now include strategic fine-tuning, mixed-source scheduling, warm-start summaries, and registry registration for strategic fine-tune artifacts

### Explicit Non-Goals For v0.3.3

- using public strategic examples as a drop-in replacement for `trajectory_steps`
- generating offline RL transition datasets from public runs without a new fidelity layer
- live web fetches inside training or evaluation loops
- final checkpoint promotion based only on public-data-backed metrics

### Exit Criteria

`v0.3.3` is complete when the repo can do all of the following:

- materialize deterministic `public_strategic_decisions` datasets from local normalized public-run artifacts
- train strategic pretraining checkpoints from those datasets with explicit build, provenance, and confidence metadata
- warm-start compatible runtime trainers from strategic pretraining checkpoints through explicit config
- compare warm-start and cold-start experiments in benchmark and registry outputs without collapsing public strategic supervision into runtime trajectory lineage

## v0.3.2

Status: `0.3.2-01` through `0.3.2-04` are complete on the current branch.

Issue breakdown:

- [0.3.2 issue index](./implementation-issues/v0.3.2/README.md)
- [0.3.2-01 SpireMeta aggregate importer and snapshot lineage](./implementation-issues/v0.3.2/0.3.2-01-spiremeta-aggregate-importer-and-snapshot-lineage.md)
- [0.3.2-02 STS2Runs public run archive contract and incremental sync](./implementation-issues/v0.3.2/0.3.2-02-sts2runs-public-run-archive-contract-and-incremental-sync.md)
- [0.3.2-03 Public run normalization and derived strategic stats](./implementation-issues/v0.3.2/0.3.2-03-public-run-normalization-and-derived-strategic-stats.md)
- [0.3.2-04 Public-source consumption in priors, predictors, and benchmark reporting](./implementation-issues/v0.3.2/0.3.2-04-public-source-consumption-in-priors-predictors-and-benchmark-reporting.md)

### Goal

Turn public community data into repo-native, incrementally refreshable artifacts without confusing aggregated card statistics with run-level supervision.

### Why This Is The Right Next Step

The current branch already has:

- a stable local-first `community_card_stats` contract for imported card pick / buy / win-delta snapshots
- policy-pack priors and benchmark alignment built on top of those imported community snapshots
- manifest-driven datasets, experiment registry outputs, predictor reports, and benchmark infrastructure

The next external-data bottleneck is no longer "can we import one snapshot?" but:

- how to ingest public aggregate stats and public run archives without collapsing them into one schema
- how to refresh those sources incrementally instead of re-pulling everything as one mutable latest snapshot
- how to derive route, event, shop, and card priors from public runs while keeping step-level BC / RL supervision anchored to our own runtime artifacts
- how to preserve source lineage, fetch cursors, dedupe logic, and freshness metadata so later reports stay reproducible

That direction is aligned with the practical shape of the current public ecosystem:

- `SpireMeta` exposes useful aggregate card statistics by character and can feed local snapshot imports cleanly
- `STS2Runs` exposes public run archives and per-run detail payloads that are better treated as a separate raw archive family
- `Spire2Stats` remains useful as a secondary reference source, but not yet as the primary integration target

### Planned Themes

- source-specific aggregate imports with snapshot lineage
- public run archive storage with resumable incremental sync
- normalized derived strategic stats from public runs
- repo-native consumption of public-source artifacts in priors, predictors, and benchmark reporting

### Delivered On The Current Branch

- repo-native `sts2-rl community import-spiremeta` command for direct `SpireMeta` aggregate imports
- `community_card_stats` snapshots now include `source-manifest.json` plus preserved raw payload files under `raw/`
- snapshot summaries now expose source kind, request count, snapshot-label histograms, raw payload roots, and fetch windows
- imported `SpireMeta` records normalize pick-rate / win-rate fields and attach card-id aliases for runtime-side consumer compatibility
- regression tests cover direct source import, lineage manifests, rate normalization, and alias-backed prior matching
- repo-native `sts2-rl public-runs sync` and `sts2-rl public-runs summary` commands for persistent `STS2Runs` archive management
- dedicated `public_run_archive` contract with cumulative index/detail JSONL files, `sync-state.json`, `source-manifest.json`, and per-session raw payload folders
- page-based incremental sync with stable run-id dedupe, sha256 duplicate tracking, detail backfill queues, retry/backoff, and resumable progress
- archive summaries now expose detail coverage, pending/failed detail counts, build and character histograms, and top-floor run slices
- regression tests cover pagination, dedupe, resumable detail backfill, and CLI summary wiring
- repo-native `sts2-rl public-runs normalize` and `sts2-rl public-runs normalized-summary` commands for turning raw archives into reusable analytical artifacts
- normalized outputs now include typed run summaries plus strategic card, shop, event, relic, encounter, and route JSONL exports
- normalized summaries now expose character/build/ascension/outcome coverage, room-type histograms, and final deck/relic coverage flags
- regression tests cover normalized export generation, missing-detail fallback behavior, and CLI normalize/summary wiring
- community prior configs now accept explicit run-derived card and route artifact paths plus freshness guards instead of only aggregate card snapshots
- policy-pack map, reward, selection, and shop decisions now surface aggregate-card vs run-derived public priors with explicit `artifact_family` provenance
- benchmark suite case and suite summaries now include public-source diagnostics so aggregate card baselines and run-derived strategic baselines remain distinguishable
- predictor calibration, ranking, and benchmark-compare reports now accept explicit public aggregate / public-run artifact paths and emit provenance, freshness, and sample-size sections

### Scope

1. SpireMeta aggregate importer and lineage

- add a source-aware importer for `SpireMeta` aggregate card stats instead of relying on manual CSV preparation
- preserve snapshot date, fetch time, request parameters, source URL, and raw payload lineage alongside canonical records
- keep the imported output compatible with the existing `community_card_stats` family

2. STS2Runs raw archive and incremental sync

- add a separate `public_run_archive` artifact family for public run list pages and per-run detail payloads
- support resumable page-based sync, stable dedupe keys, and detail backfill without rewriting prior snapshots
- persist cursor, freshness, and fetch-manifest metadata so repeated syncs are auditable

3. Public run normalization and derived stats

- normalize archived public runs into typed repo-side records for route, deck, relic, reward, shop, and encounter summaries
- derive strategic statistics and benchmark slices from those normalized runs
- keep explicit separation between run-level public summaries and our step-level runtime datasets

4. Consumption in priors, predictors, and reporting

- let priors consume derived public-run card / shop / route signals with clear provenance
- let predictors and benchmarks report alignment against public-run-derived slices and public aggregate baselines
- expose freshness, source coverage, and sample-size diagnostics in summaries so external signals can be trusted or ignored explicitly

### Reference-Driven Decisions

- `SpireMeta` supports extending `community_card_stats` with source-specific imports because it already exposes stable aggregate card endpoints and card metadata.
- `STS2Runs` supports introducing a separate raw archive family because its public API exposes paged run listings and per-run detail payloads that are richer than aggregate card snapshots.
- `Spire2Stats` supports keeping a secondary-source option in scope later because its frontend clearly maps uploaded `.run` payloads into reusable run fields, but its public structured API surface is less direct.

### Explicit Non-Goals For v0.3.2

- using public runs as the primary behavior-cloning or RL supervision corpus
- replacing local runtime collection with website scraping
- scraping live websites in the online control loop
- building a generic multi-source crawler before source-specific contracts exist

### Exit Criteria

`v0.3.2` is complete when the repo can do all of the following:

- import `SpireMeta` aggregate card stats directly into lineage-rich local snapshots
- sync `STS2Runs` incrementally into a deduplicated public run archive with resumable manifests and detail backfill
- derive normalized strategic summaries from public runs without conflating them with step-level runtime trajectories
- consume public aggregate and run-derived signals in priors, predictors, and benchmark/reporting flows with explicit freshness and provenance

## v0.3.0

Status: `0.3.0-01` through `0.3.0-03` are complete on the current branch.

Issue breakdown:

- [0.3.0 issue index](./implementation-issues/v0.3.0/README.md)
- [0.3.0-01 Shadow combat encounter snapshots and dataset contract](./implementation-issues/v0.3.0/0.3.0-01-shadow-combat-encounter-snapshots-and-dataset-contract.md)
- [0.3.0-02 Shadow rollout and planner-search harness](./implementation-issues/v0.3.0/0.3.0-02-shadow-rollout-and-planner-search-harness.md)
- [0.3.0-03 Search-guided evaluation and benchmark integration](./implementation-issues/v0.3.0/0.3.0-03-search-guided-evaluation-and-benchmark-integration.md)

### Goal

Add a faster inner loop for algorithm research without abandoning the real runtime.

### Planned Themes

- limited combat simulator or shadow environment for selected encounters
- planner and search integration against the shadow environment
- deeper model families once data and evaluation are already stable

### Delivered On The Current Branch

- manifest-built `shadow_combat_encounters` datasets from trajectory logs
- typed encounter-start records with start snapshots, strategic context, action traces, and terminal combat outcomes
- `combat_id` split grouping support for encounter-level train/validation/test separation
- `encounters.jsonl` / `encounters.csv` exports plus encounter-family and snapshot-coverage summaries
- repo-native `sts2-rl shadow combat-eval` and `sts2-rl shadow combat-compare` commands over stored encounter datasets
- shadow eval / compare summaries that report first-action-match, trace-hit, agreement, and candidate-advantage metrics
- offline harness tests covering snapshot replay, skipped-encounter handling, CLI execution, and summary loading
- benchmark `compare` cases can now attach shadow encounter datasets and emit sidecar shadow compare artifacts
- promotion gates can consume shadow comparable-count, candidate-advantage, and logged-action delta checks alongside live strategic metrics
- benchmark suite summaries and registry snapshots now surface shadow compare coverage and shadow delta signals

## v0.2.2

Status: complete on the current branch. `0.2.2-01` through `0.2.2-03` are complete on the current branch.

Issue breakdown:

- [0.2.2 issue index](./implementation-issues/v0.2.2/README.md)
- [0.2.2-01 Runtime normalization and anchored starts](./implementation-issues/v0.2.2/0.2.2-01-runtime-normalization-and-anchored-starts.md)
- [0.2.2-02 Replay and compare artifact hardening](./implementation-issues/v0.2.2/0.2.2-02-replay-and-compare-artifact-hardening.md)
- [0.2.2-03 Eval-driven checkpoint promotion](./implementation-issues/v0.2.2/0.2.2-03-eval-driven-checkpoint-promotion.md)

### Goal

Turn `v0.2.1` from a replay-capable stack into an operationally anchored evaluation loop that can recover live runtimes, compare checkpoints from controlled starts, and promote checkpoints inside schedules using explicit evaluation artifacts instead of training-only heuristics.

### Scope

1. Runtime normalization and anchored starts

- add a general runtime normalization helper that can recover arbitrary live screens to `main_menu` or `character_select`
- expose normalization through the CLI for real instance configs instead of keeping it buried inside replay
- prefer direct exits such as `return_to_main_menu`, `close_main_menu_submenu`, `dismiss_modal`, `confirm_modal`, and `abandon_run` before falling back to heuristic progression

2. Replay and compare artifact hardening

- replace replay-only `prepare_main_menu` assumptions with explicit `prepare_target`
- persist normalization reports into replay and compare iteration artifacts
- make replay and checkpoint comparison summaries expose normalization stop reasons and strategy histograms

3. Eval-driven checkpoint promotion

- extend schedules with `checkpoint_source=best_eval`
- compare a session's `latest` checkpoint against `best` using the live evaluation harness
- persist promotion artifacts and the chosen resume checkpoint into schedule logs and summaries
- define explicit fallback behavior for missing best checkpoints, failed comparisons, and ties

### Delivered On The Current Branch

- `src/sts2_rl/runtime/normalize.py` adds runtime normalization reports plus direct-exit-first recovery.
- `sts2-rl instances normalize` can normalize one or more configured instances and writes per-instance JSON reports plus a summary artifact.
- replay and checkpoint comparison flows now accept `--prepare-target` with backward-compatible `--prepare-main-menu` handling.
- replay and compare iteration artifacts now include `prepare_target`, `normalization_report`, and anchored start payloads.
- schedule runs now support `checkpoint_source=best_eval` and emit promotion artifacts under `promotions/`.
- schedule summaries now record the selected checkpoint label, selection mode, comparison deltas, and fallback reasons.

### Explicit Non-Goals For v0.2.2

- introducing a new learning algorithm such as PPO or APPO
- moving transport away from the existing `STS2-Agent` HTTP bridge
- simulator-first, headless, or savestate execution
- predictor integration into online control

### Exit Criteria

`v0.2.2` is complete when the repo can do all of the following:

- normalize a live runtime to `main_menu` or `character_select` through a repo-native command
- run replay and checkpoint comparison from explicit normalized targets while writing normalization artifacts
- select the next schedule checkpoint via explicit latest-vs-best evaluation and persist the selection rationale

## v0.2.1

Status: complete on the current branch. `0.2.1-01` through `0.2.1-05` are complete on the current branch.

Issue breakdown:

- [0.2.1 issue index](./implementation-issues/v0.2.1/README.md)
- [0.2.1-01 Episode lifecycle controller](./implementation-issues/v0.2.1/0.2.1-01-episode-lifecycle-controller.md)
- [0.2.1-02 Replay and determinism harness](./implementation-issues/v0.2.1/0.2.1-02-replay-and-determinism-harness.md)
- [0.2.1-03 Predictor/value-model bootstrap](./implementation-issues/v0.2.1/0.2.1-03-predictor-value-model-bootstrap.md)
- [0.2.1-04 Heuristic collector strengthening](./implementation-issues/v0.2.1/0.2.1-04-heuristic-collector-strengthening.md)
- [0.2.1-05 DQN hardening](./implementation-issues/v0.2.1/0.2.1-05-dqn-hardening.md)

### Historical Summary

`v0.2.1` established the unattended baseline that `v0.2.2` builds on:

- episode lifecycle control across collect, train, eval, and schedules
- replay and determinism artifacts
- predictor dataset extraction and baseline training
- stronger non-combat heuristics
- Double DQN, n-step returns, prioritized replay, and richer checkpoint metadata

### Reference-Driven Decisions

These repos continue to inform the current direction:

- `CommunicationMod` and `spirecomm`: keep the bridge layer separate from agent logic
- `runlogger`: structured run logging matters, so new features extend the recorder
- `determinismfix`: determinism must be measured explicitly
- `SlayTheSpireFightPredictor`: predictor branches deserve clean exported data
- `bottled_ai`: heuristics remain useful even after RL exists
- `SuperFastMode`: throughput belongs in the runtime baseline, not a repo-side rewrite
- `SeedSearch`: headless and savestate work are valid, but still not the current patch-line priority
