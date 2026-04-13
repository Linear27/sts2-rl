# v0.3.5 Issues

This directory turns roadmap `v0.3.5` into execution-ready issues.

Status:

- `0.3.5-01` complete on the current branch

## Order

1. `0.3.5-01` Custom-run action-space and contract-native configuration

## Critical Path

- `0.3.5-01` lands first because `CUSTOM_RUN` is already a repo-owned preparation path, but the repo was still leaving runtime-supported configuration descriptors partially unsupported and configuring modifier bundles through incremental toggles.

## Guardrails

- runtime action names, payload fields, and parameter semantics remain owned by `STS2-Agent`
- this repo must consume those runtime descriptors directly instead of re-inventing parallel configuration semantics
- custom-run preparation should fail loudly when required runtime descriptors are unavailable instead of silently degrading to partial behavior
- capability diagnostics should not report repo-side unsupported descriptors for runtime actions that this repo claims to support

## Issues

- [0.3.5-01 Custom-Run Action-Space And Contract-Native Configuration](./0.3.5-01-custom-run-action-space-and-contract-native-configuration.md)
