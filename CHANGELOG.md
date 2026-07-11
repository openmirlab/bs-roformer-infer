# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
