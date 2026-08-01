# Changelog

All notable changes to this project are documented in this file.

## [0.1.6] - 2026-08-01

### Added
- **`output_format` parameter on `run_folder`/`BSRoformerSession.infer`/`separate_folder`**
  (default: `"wav_float32"`, unchanged behavior). Accepts `"wav_float32"` (`.wav`,
  `subtype="FLOAT"`, today's behavior), `"wav_s16"` (`.wav`, `subtype="PCM_16"`), or
  `"flac16"` (`.flac`, `subtype="PCM_16"` -- FLAC has no float subtype in
  soundfile/libsndfile, by design, not a limitation). The CLI gained a matching
  `--output_format` flag (same default, same choices). Single source of truth for the
  suffix/subtype mapping is `inference._OUTPUT_FORMAT_WRITERS`.

## [0.1.5] - 2026-07-12

Re-hosted the 9 registry models whose TRvlvr fallback URLs were dead (404,
audited 2026-07-12; see #2). All 9 now resolve to live, sha256-cross-verified
mirrors via `data/overrides.json`, with hashes recorded in
`data/checksums.json`. Provenance and re-verification method below (constitution
art. 4: "provenance recorded — where the checkpoint came from, sha256, who
trained it").

### Fixed
- **9 dead checkpoint/config URLs re-hosted** (all cross-verified sha256
  against at least two independently-hosted copies before being written to
  `overrides.json`):
  - **BS-Roformer-De-Reverb** (`deverb_bs_roformer_8_384dim_10depth.ckpt`,
    `9c38653aaa5e49f2f7b84dd3be2b6b679e0cbea23978e6b48389ee6f0a914768`,
    361,499,604 bytes) — checkpoint from Politrees/UVR_resources, verified
    byte-identical to the author's anvuew/dereverb_bs_roformer and to
    Sucial/MSST-WebUI copies.
    **Config uses the author's file (anvuew/dereverb_bs_roformer,
    `archive/deverb_bs_roformer_8_384dim_10depth.yaml`,
    `a87cf93b36b9a20d25a9cc4f78a2541ea0033988e7b6c38dcf0029e9290af816`,
    2,358 bytes), NOT Politrees' similarly-named
    `config_bs_roformer_deverb_8_384dim_10depth.yaml`
    (`57c7d6b6742e2dc64d091892ae6fc1160410b365c39d1db9c9d1f772f3a4d8ce`).
    The two configs diverge on exactly one load-bearing key: the author's has
    `stft_hop_length: 512`, Politrees' has `stft_hop_length: 441` (a
    UVR5-specific patch, not the training-config value this package's
    inference path needs — `stft_hop_length` is in the model-construction
    allowlist, so this silently degrades output rather than erroring.
    **Do not "fix" this back to the Politrees file** — that is the trap, not
    a typo.**
  - **Chorus Male-Female by Sucial**
    (`model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt`,
    `123c00786bdbc6bd462dddb35cd21fd6ae99ab8319f93f63a8abc1012e593d94`,
    527,121,477 bytes) and its config
    (`config_chorus_male_female_bs_roformer.yaml`,
    `363ef6bef5f0fd89b69e6e0f90dcc102dec6aaeb6e55790cae1796bebdf5097e`,
    1,847 bytes) — from Politrees/UVR_resources.
  - **Instrumental Resurrection by unwa**
    (`bs_roformer_instrumental_resurrection_unwa.ckpt`,
    `16311025a5133ae6411760ccfe9e3e66b31a01d9d8bec0a03fa7ec4bedac7a15`,
    204,483,033 bytes) and its config
    (`config_bs_roformer_instrumental_resurrection_unwa.yaml`,
    `0c67e988e1d608a4d3414f602dd3593ec927913c60618d1ff981baec40455cb0`,
    2,632 bytes) — from Politrees/UVR_resources (hosted there under the
    filename `model_BandSplit-Roformer_Resurrection_Instrumental_by-Unwa.ckpt`),
    cross-verified against the pcunwa/BS-Roformer-Resurrection author repo.
  - **Male-Female by aufr33**
    (`bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt`,
    `3cf11736d1b42a11ae55d8299316585921477dd2a671b24b663660846ca9861b`,
    527,119,779 bytes) — no author HF account exists for aufr33; from
    Politrees/UVR_resources, cross-verified byte-identical against
    Sucial/MSST-WebUI's copy (2-source agreement). Shares the Chorus model's
    config.
  - **Vocals by Gabox** (`bs_roformer_vocals_gabox.ckpt`,
    `18d58efe5e949e70fab11b875329af6d06ef11ccc29574bfe943fb57cc827f38`,
    639,254,584 bytes) and its config
    (`config_bs_roformer_vocals_gabox.yaml`,
    `2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f`,
    2,273 bytes) — from Politrees/UVR_resources (hosted there as
    `bs_roformer_voc_gabox.ckpt`), cross-verified against the
    GaboxR67/BSRoformerVocTest author repo.
  - **Vocals Resurrection by unwa**
    (`bs_roformer_vocals_resurrection_unwa.ckpt`,
    `9dbfe5cb572e4ed32a15ec727d7bd06c8d7aba97509e6fda5bc008bb1e0b2dd5`,
    204,510,749 bytes) and its config
    (`config_bs_roformer_vocals_resurrection_unwa.yaml`,
    `79c65b6158c8e9236c9f02247cbb2fb6eb0007c42c47799e9de8e86f474556c6`,
    2,633 bytes) — from Politrees/UVR_resources (hosted there as
    `model_BandSplit-Roformer_Resurrection_Vocals_by-Unwa.ckpt`).
  - **Vocals Revive by unwa** (`bs_roformer_vocals_revive_unwa.ckpt`,
    `f1d7e4bfdfef07c6b2bc1d65283a7d03c3c38f8c7dbc8d729b785f93c8b8699a`,
    639,326,600 bytes) and its config
    (`config_bs_roformer_vocals_revive_unwa.yaml`,
    `6b9d5fb6aeda6b0941f937e4e4883643e3187331ff403fb820c7aa6af4b02dbb`,
    2,382 bytes) — from Politrees/UVR_resources (hosted there as
    `bs_roformer_revive_by_unwa.ckpt`), cross-verified against the pcunwa
    author repo (`bs_roformer_revive.ckpt`).
  - **Vocals Revive V2 by unwa** (`bs_roformer_vocals_revive_v2_unwa.ckpt`,
    `58098850c882a7472dad39f99fb8040ce6eaafe671cfe9881d89aea276bbb5f5`,
    639,326,600 bytes) — from Politrees/UVR_resources (hosted there as
    `bs_roformer_revive_v2_by_unwa.ckpt`). Shares the Revive config.
  - **Vocals Revive V3e by unwa** (`bs_roformer_vocals_revive_v3e_unwa.ckpt`,
    `1b0751b9a15c591407c3b77f08eb4ad3005e42e96051f3f2b39760f1130c467b`,
    639,326,600 bytes) — from Politrees/UVR_resources, hosted there as
    `bs_roformer_revive_v3_by_unwa.ckpt` **(note the filename drops the "e" —
    this is the same file, verified byte-identical against the pcunwa author
    repo's `bs_roformer_revive3e.ckpt`; the missing "e" is a Politrees
    naming quirk, not a different model)**. Shares the Revive config.
- **Source-selection policy note** (flagged per constitution art. 8 — a
  policy-level call, not a quiet bundle): all 9 checkpoints above route
  through **Politrees/UVR_resources** uniformly rather than each model's
  original author account. Politrees is the one mirror with a fully
  enumerated, hash-verified tree covering every affected model; individual
  author-repo URLs for 7 of the 9 models were not independently re-derived
  in this pass (only anvuew's de-reverb config and the Politrees/Sucial
  cross-checks were). This satisfies constitution art. 4's "at minimum a
  mirror" bar but not its "canonical source is openmirlab's own HF account"
  ideal — migrating these to an openmirlab-controlled HF mirror remains
  open work (tracked in #2).

### Verification
- Re-ran the package's real `download_model_assets()` code path (not a
  monkeypatch) against the new `overrides.json`/`checksums.json` on disk for
  the De-Reverb model plus two spot-checks (Chorus Male-Female, Vocals
  Revive V3e) — all three downloaded and sha256-verified successfully.
- Full test suite: 27 passed, 19 deselected (unchanged from 0.1.4 baseline).

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
- **`.github/workflows/test.yml`**: push/PR-triggered CI, closing a gap found
  in an org-wide audit — the only prior workflow was `publish.yml`, which
  builds/tests on a single pinned Python (3.10) at release time only, so
  nothing ran between releases. Matrix covers Python 3.10-3.13 (all four
  verified locally: 27 passed, 19 deselected — the `network`-marked tests
  that hit real weight hosts, already excluded by `addopts = "-m 'not
  network'"`). A `build` job (needs `test`) does the wheel-from-sdist smoke
  test: `python -m build`, install the wheel into a clean venv, import
  `bs_roformer` and touch `BSRoformer`, and confirm the wheel bundles no
  `.ckpt` weight file (constitution art. 7).

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
## Unreleased

- Added the additive `BSRoformerSession` lifecycle facade and package-owned
  `config/checkpoints.toml` checkpoint metadata.
