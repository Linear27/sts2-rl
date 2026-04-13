# sts2-rl

`sts2-rl` is a Windows-first research and engineering workspace for training, evaluating, and benchmarking Slay the Spire 2 agents through the local `STS2-Agent` HTTP API.

This repo is not a game mod and not a game install mirror. It is the repo-side control stack around a live runtime: typed state ingestion, action-space materialization, trajectory capture, dataset building, training, replay/compare flows, and benchmark/reporting infrastructure.

## Why This Repo Exists

Most Slay the Spire agent projects collapse runtime glue, policy logic, and experiment code into one pile. This repo keeps them separate:

- `STS2-Agent` owns the authoritative runtime state and action contract.
- This repo owns learning code, evaluation flows, datasets, orchestration, and repo-side capability diagnostics.
- Game binaries, `.pck` assets, provisioned runtime copies, generated datasets, and experiment artifacts stay out of git.

That split matters because it keeps runtime contract changes explicit instead of hiding them inside prompt heuristics or one-off scripts.

## Current Status

- Current branch status: `v0.3.5-01` complete
- License: [Apache-2.0](./LICENSE)
- Roadmap source of truth: [docs/roadmap.md](./docs/roadmap.md)
- Implementation history and issue breakdown: [docs/implementation-issues](./docs/implementation-issues)

Completed roadmap lines currently tracked here:

- `v0.3.5`
- `v0.3.4`
- `v0.3.3`
- `v0.3.2`
- `v0.3.1`
- `v0.3.0`
- `v0.2.5`
- `v0.2.4`
- `v0.2.3`
- `v0.2.2`
- `v0.2.1`

## What It Can Do

- ingest typed runtime state and legal actions from `STS2-Agent`
- expand repo-side candidate actions for combat and non-combat control
- normalize live runtimes back to safe starting screens
- prepare deterministic `CUSTOM_RUN` starts for fixed-seed evaluation
- collect trajectory logs and derived combat / strategic artifacts
- build datasets for behavior cloning, offline RL, predictor training, and strategic pretraining
- run behavior cloning, offline CQL, combat DQN, and strategic fine-tuning flows
- replay, compare, benchmark, and promote checkpoints with structured diagnostics
- ingest community stats and public-run archives as external initialization / analysis inputs

## Requirements

- Python `3.11+`
- `uv`
- A local Slay the Spire 2 install
- A compatible `STS2-Agent` setup
- Windows environment for the current bring-up scripts

## Architecture At A Glance

```text
Slay the Spire 2 runtime copy
  -> STS2-Agent HTTP API
  -> typed client / env wrapper
  -> candidate-action expansion
  -> collection / replay / benchmark flows
  -> datasets / training / evaluation / registry artifacts
```

The repo boundary in one sentence:

- runtime contract and state authority belong to `STS2-Agent`
- decision logic, training, datasets, evaluation, and repo-side diagnostics belong here

For more detail, see [docs/architecture.md](./docs/architecture.md).

## Quick Start

1. Install dependencies:

```powershell
uv sync --extra dev
```

2. Read the bring-up docs:

- [docs/bringup.md](./docs/bringup.md)
- [docs/architecture.md](./docs/architecture.md)

3. Prepare a local instance config:

- Start from [configs/instances/local.single.example.toml](./configs/instances/local.single.example.toml).
- Replace the placeholder game paths with absolute paths on your machine.
- If you do not want local edits tracked, copy it to a gitignored `*.private.toml` or `*.local.toml` file and use that instead.

4. Verify the repo:

```powershell
uv run pytest -q
```

5. Verify the runtime bridge:

```powershell
uv run sts2-rl instances status --config configs/instances/local.single.example.toml --timeout-seconds 5
uv run sts2-rl benchmark health --base-url http://127.0.0.1:8080
```

## Common Tasks

### Runtime bring-up

```powershell
uv run sts2-rl instances plan --config configs/instances/local.single.example.toml
uv run sts2-rl instances provision --config configs/instances/local.single.example.toml
uv run sts2-rl instances normalize --config configs/instances/local.single.example.toml --target main_menu
```

### Collect one smoke rollout

```powershell
uv run sts2-rl collect rollouts --config configs/instances/local.single.example.toml --output-root data/trajectories --session-name smoke-v2 --max-steps-per-instance 0 --max-runs-per-instance 1
```

### Train and evaluate a smoke combat DQN

```powershell
uv run sts2-rl train combat-dqn --base-url http://127.0.0.1:8080 --output-root artifacts/training --session-name smoke-dqn --max-env-steps 0 --max-runs 1
uv run sts2-rl eval combat-dqn --base-url http://127.0.0.1:8080 --checkpoint-path artifacts/training/smoke-dqn/combat-dqn-checkpoint.json --output-root artifacts/eval --session-name eval-dqn --max-env-steps 0 --max-runs 1
```

### Run behavior cloning from a prepared dataset

```powershell
uv run sts2-rl train behavior-cloning --dataset data/trajectory/live-v023 --output-root artifacts/behavior-cloning --session-name bc-live --epochs 40 --live-base-url http://127.0.0.1:8080
```

More example manifests and command shapes live under:

- [configs/training](./configs/training)
- [configs/benchmarks](./configs/benchmarks)
- [configs/datasets](./configs/datasets)
- [configs/jobs](./configs/jobs)
- [configs/dags](./configs/dags)

## Repository Layout

- `src/sts2_rl/`: Python package
- `tests/`: unit and regression coverage
- `configs/`: tracked templates and example manifests
- `docs/`: roadmap, bring-up, architecture, and issue breakdowns
- `scripts/`: PowerShell helpers for local Windows workflows
- `runtime/`: local runtime copies, kept out of git
- `data/`: local datasets and templates
- `artifacts/`: generated checkpoints, reports, and summaries

## Important Assumptions

- Reference game copies and clean baselines live outside this repo.
- Default endpoint examples use `http://127.0.0.1:8080`.
- Fixed-seed live evaluation uses explicit custom-run contracts, not the trainer RNG seed.
- Public strategic data is treated as initialization / analysis evidence, not control-equivalent runtime supervision.

See [docs/custom-seed-benchmark-protocol.md](./docs/custom-seed-benchmark-protocol.md) for the current fixed-seed benchmark contract.

## Documentation

- [docs/roadmap.md](./docs/roadmap.md)
- [docs/bringup.md](./docs/bringup.md)
- [docs/architecture.md](./docs/architecture.md)
- [docs/custom-seed-benchmark-protocol.md](./docs/custom-seed-benchmark-protocol.md)
- [docs/public-release-checklist.md](./docs/public-release-checklist.md)
- [CONTRIBUTING.md](./CONTRIBUTING.md)

## Public Repo Notes

- The repo is publishable from a structure standpoint; no large-scale tree reorganization is required first.
- Before the first public push, walk [docs/public-release-checklist.md](./docs/public-release-checklist.md) once from a clean-clone mindset.
