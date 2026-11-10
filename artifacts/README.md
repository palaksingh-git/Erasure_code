# Full local project copy

This repository stores the source code and configuration in Git. The generated
images, evaluation tables, classifier weights, checkpoints, original Windows
virtual environment, and source Git history are packaged in the release tagged
`full-project-artifacts-2026-11-10`. The manifest records sizes and SHA-256
checksums. Checkpoint payloads are split into parts below GitHub's per-asset
size limit; identical checkpoint payloads are uploaded once and restored to
all their original paths.

From a fresh clone, restore the generated artifacts with:

```powershell
python scripts/restore_release_artifacts.py
```

To also extract the original Windows virtual environment and download the
source-history bundle:

```powershell
python scripts/restore_release_artifacts.py --include-venv --include-history
```

The source-history bundle downloads into `results/migration_download/` and
does not replace this repository's Git history. The copied virtual environment
contains machine-specific paths; installing fresh dependencies from
`pyproject.toml` is recommended on another machine. The Stable Diffusion 1.4
base model cache lives outside the project and is downloaded by Diffusers on
first use, as described in the main README.
