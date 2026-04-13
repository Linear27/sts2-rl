# Contributing

## Scope

This repo owns training, evaluation, data contracts, and runtime orchestration for `STS2-Agent`-backed Slay the Spire 2 experiments. It does not own the game install, game assets, or the runtime HTTP contract itself.

## Before opening a change

- Read [README.md](./README.md) for repo scope and bring-up expectations.
- Read [docs/roadmap.md](./docs/roadmap.md) and the relevant implementation issue under [docs/implementation-issues](./docs/implementation-issues/).
- Keep repo-side ownership explicit. If a change depends on new runtime fields or actions, the contract belongs in `STS2-Agent`, not in guessed repo-side behavior.

## Development expectations

- Use Python 3.11+ and `uv`.
- Install dev dependencies with `uv sync --extra dev`.
- Run targeted tests for the files you changed before opening a PR.
- Keep local runtime paths, user data, and machine-specific configs out of git. Use `*.private.toml` or `*.local.toml` for local overrides.

## Change shape

- Prefer issue-sized changes over mixed refactors.
- Update docs when behavior, contracts, or workflow expectations change.
- Add or extend regression tests for new action-space coverage, dataset contracts, or benchmark/reporting semantics.
- Do not commit `runtime/`, `artifacts/`, generated datasets, or local user-data snapshots.

## Pull requests

- Explain the problem, repo-vs-runtime ownership boundary, and the exact acceptance criteria being closed.
- Include the test command(s) you ran and the result.
- Call out any follow-up work explicitly instead of leaving silent partial states.
