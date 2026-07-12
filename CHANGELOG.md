# Changelog

All notable changes to this project are documented in this file.

## [0.1.4] - 2026-07-12

Weights-UX campaign (org Weights UX contract, constitution art. 4): real
integrity verification, true auto-download, and a configurable weights folder.

### Added
- **sha256 verification wired into downloads**: `data/checksums.json` records
  the known-good sha256 + size for every asset with a live download URL
  (BS-Rofo-SW-Fixed.ckpt:
  `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e`,
  699,412,152 bytes — obtained from the Hugging Face LFS pointer and
  cross-verified against two independently downloaded local copies).
  `download_file`/`verify_file_integrity` now check it after every download;
  a mismatch deletes the file and retries instead of keeping a corrupt
  checkpoint. The previously dead `get_file_hash()` helper is now the actual
  verification path. Assets without a recorded hash fall back to the old
  non-empty check and say so.
- **Auto-download on first use**: `bs-roformer-infer` no longer requires
  `--model_path`/`--config_path` — when omitted, the registry model selected
  via the new `--model` flag (default: BS-RoFormer-SW) is resolved from the
  models directory and downloaded (sha256-verified) if missing, via the new
  public `ensure_model_assets()` API (also exported from the package root).
  Explicit paths still win and skip auto-resolution.
- **Configurable weights folder**: downloads now default to
  `~/.cache/bs-roformer-infer/` instead of the CWD-relative `./models`.
  Overridable via the `BS_ROFORMER_MODELS_PATH` env var, the inference CLI's
  `--models_dir`, the download CLI's `--output-dir`, or the API's
  `models_dir=` argument. A legacy `./models` directory is still searched as
  a read fallback so pre-0.1.4 downloads keep working without re-fetching.
- `tests/test_weights_ux.py`: offline unit tests for the hash-verification
  wiring (including a fake-response download that must reject a wrong sha256
  and delete the file) and the models-dir resolution order.

### Changed
- Version is now single-sourced from `src/bs_roformer/__about__.py` via
  hatch's dynamic version; `pyproject.toml` and `__init__.py` no longer carry
  duplicate literals.
- README: weights section rewritten to state the auto-download behavior, the
  manual download recipe (exact URL + sha256 + target path), and the folder
  resolution order truthfully. The previous "integrity verification" feature
  claim described code that never ran; it is now accurate.

### Known issue (documented, not fixed here)
- Liveness audit 2026-07-12: **8 of the 9 registry models are currently
  unavailable** — their TRvlvr fallback URLs 404 on both checkpoint and
  config. Only the default BS-RoFormer-SW model (enerjazzer mirror) is
  downloadable today; the dead entries therefore have no recorded checksums
  yet. Finding live mirrors is tracked as follow-up work.

### Removed
- Google Colab quickstart notebook (`notebooks/quickstart_colab.ipynb`) and
  the README's Colab badge/section — maintaining a separate notebook
  environment alongside the PyPI package was more upkeep than the audience
  justified.

## [0.1.3] - 2026-07-11

### Fixed
- Declare the missing `packaging` dependency — `attend.py` imports it for the
  torch version check, but it was never in `pyproject.toml`; clean environments
  could fail with `ModuleNotFoundError` (#1, thanks @derVedro).

## [0.1.2] - 2026-07-11

- **Fix**: model weights URL — the jarredou HuggingFace account behind the default
  BS-RoFormer-SW checkpoint/config was deleted (discovered 2026-06), 404ing
  downloads for every user of the recommended default model. Repointed
  `overrides.json` to the verified `enerjazzer/BS-ROFO-SW-Fixed` mirror
  (sha256 `24e7d35e…775916e`, 699,412,152 bytes) that was already on `main`
  (557f791) but never shipped to PyPI.
- **Fix**: the packaged BS-RoFormer-SW config was bundled under a different
  filename than `BSModel.config` expected, so the local copy was never matched
  and every install silently re-fetched the config over the network. Renamed
  the bundled file to line up; added a test proving the packaged-config path
  now hits without touching the network.
- **Add**: URL regression tests (`tests/test_download.py`) that lock the SW
  model's checkpoint/config overrides to the enerjazzer mirror, so a future
  silent revert to a dead URL fails tests instead of shipping quietly.
- **Add**: `tools/check_weights_liveness.py` + `tests/test_weights_liveness.py`
  — a `pytest.mark.network`-gated liveness check (skipped by default; run with
  `pytest -m network`) that HEADs every registry download URL.
- **Add**: file-top nav headers (title + rationale + `Reads:` deps) across all
  source modules and the test suite, per the org's navigability convention.
- **Chore**: removed a handful of ruff-flagged unused imports and extraneous
  f-string prefixes; no behavior change.

## [0.1.1] - 2026-06

- See git history prior to this file's creation.
