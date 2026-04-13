# Bring-up Notes

## Runtime constraints

- `STS2-Agent` exposes a local HTTP API by default on `http://127.0.0.1:8080`.
- The main endpoints used by this repo are `GET /health`, `GET /state`, `GET /actions/available`, and `POST /action`.
- `STS2-Agent` supports overriding the API port through `STS2_API_PORT`.
- Provisioned runtime copies remain disposable; the clean baseline stays outside the repo.

## What the repo owns

- Per-instance runtime planning, provisioning, manifests, and health checks
- Windows launch plans around `scripts/start-sts2-instance.ps1`
- Typed HTTP client and environment wrapper
- Collection, training, and evaluation CLIs
- Unified trajectory schema v2 with:
  - `session_started/session_finished`
  - `run_started/run_finished`
  - `floor_started/floor_finished`
  - `combat_started/combat_finished`
  - `step`
- Per-session `summary.json`
- Per-session `combat-outcomes.jsonl`

## Current Windows workflow

1. Keep Steam logged in.
2. Provision one repo-owned runtime root per instance from the clean baseline.
3. Seed each instance user dir from an already-approved template so mod warnings are already accepted.
4. Launch copied executables directly with a unique `STS2_API_PORT`.

Important details:

- Provisioning writes an `override.cfg` so each Godot instance uses a distinct `user://` root.
- Fresh user dirs still require one manual source instance to accept the mod warning and run `unlock all`.
- `bootstrap-user-data` snapshots that approved source dir into `data/user-data-templates/golden/`; that directory is local-only and should not be committed.
- If the default renderer is unstable on your machine, prefer `opengl3`.
- The low-thermal default is `configs/instances/local.single.example.toml`.
- `scripts/run-local-single.ps1` prefers `configs/instances/local.single.private.toml` when it exists and otherwise falls back to the tracked example.

## Example low-thermal flow

```powershell
uv run sts2-rl instances status --config configs/instances/local.single.example.toml --timeout-seconds 5
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 start
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 collect -SessionName smoke-v2 -MaxStepsPerInstance 0 -MaxRunsPerInstance 1
```

Budget notes:

- `0` disables a budget.
- Collection supports `MaxStepsPerInstance`, `MaxRunsPerInstance`, and `MaxCombatsPerInstance`.
- Train and eval support `MaxEnvSteps`, `MaxRuns`, and `MaxCombats`.
- Replay suites support `RepeatCount`, `MaxEnvSteps`, `MaxRuns`, and `MaxCombats`.
- Schedule supports `MaxEnvSteps`, `MaxRuns`, and `MaxCombats` as per-session budgets.

## DQN workflow

### Train

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 train -SessionName smoke-dqn -MaxEnvSteps 0 -MaxRuns 1
```

### Resume

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 train -SessionName resume-dqn -ResumeFrom .\artifacts\training\smoke-dqn\combat-dqn-checkpoint.json -MaxEnvSteps 0 -MaxRuns 1
```

### Eval

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 eval -SessionName eval-dqn -CheckpointPath .\artifacts\training\smoke-dqn\combat-dqn-checkpoint.json -MaxEnvSteps 0 -MaxRuns 1
```

### Replay

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 replay -SessionName replay-dqn -CheckpointPath .\artifacts\training\smoke-dqn\combat-dqn-checkpoint.json -RepeatCount 3 -MaxEnvSteps 0 -MaxRuns 1
```

### Schedule

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local-single.ps1 schedule -SessionName schedule-dqn -ResumeFrom .\artifacts\training\smoke-dqn\combat-dqn-checkpoint.json -MaxSessions 8 -MaxEnvSteps 0 -MaxRuns 1 -CheckpointEveryRlSteps 25
```

## Session artifacts

### Collection session

- `inst-01.jsonl`
- `inst-01-summary.json`
- `inst-01-combat-outcomes.jsonl`

### Training session

- `combat-train.jsonl`
- `summary.json`
- `combat-outcomes.jsonl`
- `combat-dqn-checkpoint.json`
- `combat-dqn-best.json`
- `checkpoints/combat-dqn-step-000123.json`

### Evaluation session

- `combat-eval.jsonl`
- `summary.json`
- `combat-outcomes.jsonl`

### Replay suite

- `replay-suite.jsonl`
- `replay-summary.json`
- `replay-comparisons.jsonl`
- `iteration-001/`, `iteration-002/`, ...

### Schedule session

- `schedule.jsonl`
- `schedule-summary.json`
- `session-001/`, `session-002/`, ...
