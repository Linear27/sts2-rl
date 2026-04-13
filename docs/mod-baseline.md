# Mod Baseline

This document defines what belongs in the clean training baseline.

## Goal

The clean baseline is the parent image for all runtime instances.

It should contain:

- The game files for a specific verified version
- `STS2-Agent`
- One animation acceleration mod
- Required dependency mods for the chosen animation mod

It should not contain:

- Telemetry-only experiment mods
- Ad hoc debugging mods unrelated to training
- Old test payloads left over from prior experiments

## STS2-Agent payload

According to the official README and build docs, the game `mods/` directory should contain:

- `STS2AIAgent.dll`
- `STS2AIAgent.pck`
- `mod_id.json`

Sources:

- https://github.com/CharTyr/STS2-Agent/blob/main/README.md
- https://github.com/CharTyr/STS2-Agent/blob/main/build-and-env.md

## Animation acceleration

For the current plan, use an existing acceleration mod rather than custom engine surgery.

`Quick Animation Mode` is a valid candidate and its Nexus page lists `RitsuLib` as a requirement.

Source:

- https://www.nexusmods.com/slaythespire2/mods/171

Because file names can vary by release package, this repo treats animation-mod validation as:

- hard requirement: none yet
- dependency hint: check for a `RitsuLib`-like payload when animation acceleration is enabled

## Recommended clean baseline workflow

1. Copy the verified reference build into a separate clean baseline directory.
2. Empty the `mods/` directory.
3. Install only:
   - `STS2-Agent`
   - chosen acceleration mod
   - required dependencies such as `RitsuLib`
4. Run `sts2-rl instances preflight`.
5. Only after preflight passes or reduces to expected warnings should the baseline be used for runtime instance creation.
