# Architecture

## Repository boundary

This repository does not live inside the Slay the Spire 2 install tree.

- Game binaries and `.pck` assets stay outside git.
- Runtime instances are disposable local copies.
- The repo stores orchestration, learning code, configs, and experiment metadata.

## System layers

### 1. Runtime layer

Responsible for local game instances.

- Provision a clean instance from a known baseline
- Assign a unique `STS2_API_PORT`
- Track process metadata and health
- Keep per-instance logs separate

### 2. Environment layer

Responsible for mapping `STS2-Agent` HTTP endpoints into a learning-friendly interface.

- Pull compact state
- Pull legal actions
- Submit chosen actions
- Normalize errors, retries, and terminal conditions

### 3. Data layer

Responsible for trajectory capture and preprocessing.

- Raw step logs in JSONL
- Processed datasets in Parquet
- Versioned schema tied to game build and mod build

### 4. Training layer

Responsible for imitation learning and online RL.

- Behavior cloning
- Policy scoring over legal actions
- PPO or APPO after a stable baseline exists

### 5. Evaluation layer

Responsible for repeatable benchmarks.

- Single-instance throughput
- Multi-instance stability
- Run-level success metrics
- Per-screen decision quality

## Initial constraints

- Use local HTTP for training, not MCP.
- Keep the reference game directory read-only.
- Validate single-instance throughput before scaling out.
- Treat shared save/config paths as an unresolved risk until verified.
