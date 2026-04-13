# Registry

`v0.2.4-04` adds a local-first experiment registry so the repo can answer "what is the current best artifact?" without manual directory inspection.

## Layout

- `artifacts/registry/registry-manifest.json`
- `artifacts/registry/experiments/<experiment-id>.json`
- `artifacts/registry/aliases.json`
- `artifacts/registry/alias-history.jsonl`
- `artifacts/registry/reports/<session>/leaderboard-summary.json`
- `artifacts/registry/reports/<session>/compare-summary.json`

Immutable experiment snapshots live under `experiments/`.
Mutable alias pointers and promotion history live outside those snapshots.

## Supported Artifact Sources

- dataset directories or `dataset-summary.json`
- behavior-cloning training runs
- offline CQL training runs
- combat DQN training runs
- policy evaluation runs with checkpoint metadata
- predictor training summaries or predictor model JSON files
- predictor report summaries
- benchmark suite summaries and benchmark case summaries

## CLI

```powershell
uv run sts2-rl registry init --root artifacts/registry
uv run sts2-rl registry register --root artifacts/registry --source artifacts/behavior-cloning/bc-live --alias best_bc
uv run sts2-rl registry register --root artifacts/registry --source artifacts/offline-cql/offline-live --alias best_offline_rl
uv run sts2-rl registry register --root artifacts/registry --source artifacts/predict-reports/live-v021-ranking --alias best_predictor
uv run sts2-rl registry list --root artifacts/registry
uv run sts2-rl registry show --root artifacts/registry --experiment best_bc
uv run sts2-rl registry alias set --root artifacts/registry --alias-name recommended_default --experiment best_bc --artifact-path-key best_checkpoint_path
uv run sts2-rl registry leaderboard --root artifacts/registry --family behavior_cloning
uv run sts2-rl registry compare --root artifacts/registry --experiment best_bc --experiment best_offline_rl
uv run sts2-rl registry promote --root artifacts/registry --alias-name recommended_default --family behavior_cloning
```

## Promotion Conventions

- `best_bc` should point to the preferred behavior-cloning checkpoint artifact.
- `best_offline_rl` should point to the preferred offline RL checkpoint artifact.
- `best_predictor` should point to the preferred predictor model or predictor report model path.
- `recommended_default` should point to the artifact currently intended as the repo's default promoted policy.

`registry promote` selects the top leaderboard row for the provided filters when `--experiment` is omitted, then records the alias update in `alias-history.jsonl`.
