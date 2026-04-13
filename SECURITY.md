# Security Policy

## Supported Scope

Security reports are accepted for the current `main` branch of this repository.

This policy covers repo-owned code and materials such as:

- Python package code under `src/`
- tracked scripts under `scripts/`
- tracked configs and documentation that affect repo-side behavior

This policy does not cover:

- Slay the Spire 2 game binaries or assets
- local runtime copies under `runtime/`
- generated local artifacts or datasets
- vulnerabilities in `STS2-Agent` itself unless the issue is caused by this repo's usage or packaging of it

If the issue belongs to `STS2-Agent` or another upstream dependency, report it to that upstream project.

## How To Report

Do not open a public issue for a security-sensitive report.

Preferred path:

- Use GitHub private vulnerability reporting or a GitHub security advisory for this repository if that option is available.

If private reporting is not available:

- Open a minimal public issue requesting a private contact channel.
- Do not include exploit details, credentials, private paths, or reproduction steps that would materially increase risk.

## What To Include

Please include:

- affected commit, branch, or file path when known
- a short description of impact
- reproduction steps
- environment details relevant to the issue
- whether the issue affects only this repo or appears to involve `STS2-Agent` / another dependency

## Response Expectations

- Reports are handled on a best-effort basis.
- Initial triage may include confirming repo ownership before any fix work starts.
- Fixes may land only on `main` rather than being backported to historical roadmap lines.

## Disclosure Guidance

- Give maintainers reasonable time to investigate and patch before public disclosure.
- If you accidentally discover committed credentials or personal data, report it immediately and avoid redistributing it further.
