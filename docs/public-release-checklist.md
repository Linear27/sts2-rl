# Public Release Checklist

Use this checklist before pushing the repo to a public remote.

## Must finish

- Confirm `.gitignore` excludes runtime copies, artifacts, datasets, local user-data snapshots, and machine-specific configs.
- Confirm `configs/` contains only safe tracked templates or examples.
- Confirm `README.md` matches the current roadmap and current bring-up flow.
- Confirm no tracked file contains personal absolute paths, tokens, or private infrastructure references.
- Choose and add a root `LICENSE` file. Do not publish without making that decision explicitly.

## Recommended

- Add `SECURITY.md` if you want a standard disclosure channel.
- Add a short release tag or initial changelog entry for the first public snapshot.
- Verify `uv sync --extra dev` and `uv run pytest -q` work from a clean clone.
- Review root docs for Windows-only assumptions so external users do not mistake the repo for a turnkey cross-platform package.

## Repo-specific notes

- `STS2-Agent` is an external dependency and owns the authoritative runtime HTTP contract.
- Game binaries, `.pck` assets, and provisioned runtime copies must stay out of git.
- `configs/instances/local.single.example.toml` is a tracked template, not a personal machine config.
