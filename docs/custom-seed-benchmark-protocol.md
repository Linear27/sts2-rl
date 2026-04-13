# Custom Seed Benchmark Protocol

## Purpose

Define a deterministic live-runtime benchmark contract for the current stage of `sts2-rl`.

This protocol exists because:

- the game supports manual Custom Mode seed entry
- our repo already distinguishes training RNG seeds from the in-game run seed

For the current branch, fixed-seed runs are treated as a benchmark runtime contract with a repo-native automated preparation path.

## Current Confirmed Facts

- `Slay the Spire 2:Custom Mode` supports manual seed entry and Ascension selection.
- the current runtime payload exposes the observed in-game seed via `run.seed` and `character_select.seed`
- the `STS2-Agent` runtime interface exposes `open_custom_run`, `set_custom_seed`, `set_custom_ascension`, and `toggle_custom_modifier`
- the repo CLI `--seed` flags are training or agent RNG seeds only; they do not set the in-game run seed
- the repo can now normalize to `main_menu`, open `Custom Mode`, apply a seed contract, and start the run before live rollout begins

## Benchmark Contract

When running deterministic benchmark sessions, the intended run contract must be:

- `run_mode = "custom"`
- `game_seed = "<seed string>"`
- `seed_source = "manual_custom_mode_ui"`
- `character_id = "<character id>"`
- `ascension = <integer>`
- `custom_modifiers = []` for the default baseline unless the benchmark explicitly requires modifiers
- `progress_profile = "unlock_all"` when using the unlocked debug profile
- `game_build = "<game version>"`
- `mod_stack = ["STS2-Agent", "..."]`

## Repo-Native Runtime Preparation

The current automated preparation path is:

1. Normalize or navigate to `main_menu`.
2. Clear any resume-state residue from `MAIN_MENU` until `open_custom_run` is available.
3. Open `Custom Mode`.
4. Select the target character when needed.
5. Set Ascension explicitly when needed.
6. Apply the target seed explicitly when needed.
7. Ensure the intended modifier set is applied.
8. Start the run.

This preparation now runs automatically for live collect / train / eval entry points when the live run contract sets `run_mode = "custom"`.

## Required Logging Semantics

For any fixed-seed benchmark run, session metadata should record the intended contract in the session `config` payload.

Minimum required fields:

- `run_mode`
- `game_seed`
- `seed_source`
- `character_id`
- `ascension`
- `custom_modifiers`
- `progress_profile`
- `benchmark_contract_id`

The existing observed fields remain authoritative runtime evidence:

- `observed_seed`
- `observed_run_seeds`
- `observed_run_seed_histogram`
- `runs_without_observed_seed`
- `config.custom_run_prepare`

## Validation Rules

A fixed-seed benchmark session is valid only if:

- every completed run reports an observed seed
- `observed_run_seed_histogram` contains exactly one seed
- that seed matches the intended `game_seed`
- the run payload shows the intended `character_id`
- the run payload shows the intended `ascension`

The session should be treated as drifted or invalid if:

- the observed seed is missing
- more than one observed seed appears in the session
- the observed seed differs from the intended seed
- the wrong character or Ascension is detected

## Practical Scope

This protocol is appropriate for:

- live smoke validation
- deterministic benchmark suites
- early-stage training on a single seed
- short seed suites for reproducible comparison

This protocol is not the final target for generalization. It is a controlled benchmark mode for the current low-parallelism phase.

## Follow-Up Engineering Work

Future automation work should target:

1. benchmark-suite manifests that declare explicit fixed-seed preparation contracts
2. replay / compare flows that can re-prepare custom runs between iterations
3. richer modifier addressing for duplicated modifier ids
4. benchmark suites keyed by explicit game seeds rather than only observed replay summaries
