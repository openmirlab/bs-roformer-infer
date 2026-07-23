# bs-roformer-infer -- CLAUDE.md

## Scope

bs-roformer-infer is an inference-only package wrapping BS-RoFormer
(Band-Split RoPE Transformer) music source separation. It reprovides the
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
architecture as a pip-installable, PyTorch-based CLI + Python API with
automatic checkpoint management: no training code, no UVR GUI dependency.

Devices preserve legacy `None` auto-selection and also accept explicit `auto`,
`cpu`, `cuda`, and `cuda:N`; explicit CUDA errors must raise. Sessions may
release and reload models, but closed sessions are terminal. `cache_info()` and
loading share the download resolver, which reads package-owned checkpoints TOML.
Given an input folder of WAV files, it produces separated stems (vocals,
drums, bass, guitar, piano, other) plus an `*_instrumental.wav` per track.
See README.md for the public API, CLI, and full model registry.

**In scope**: inference (forward pass) only; a 24-model registry
(`src/bs_roformer/config/checkpoints.toml`) spanning multi-stem, 53-stem mega,
four-stem, vocals, karaoke, instrumental, and de-reverb checkpoints; sha256-verified auto-download with a
configurable-dir UX contract (explicit arg > `$BS_ROFORMER_MODELS_PATH` >
`~/.cache/bs-roformer-infer`, legacy `./models` honored as a read fallback);
manual/offline install path; a download CLI (`bs-roformer-download`)
independent of the inference CLI.

**Out of scope, forever**: training/fine-tuning code, the UVR GUI itself,
hosting or mirroring checkpoint bytes in this repo's git history (weights are
always fetched at runtime -- see README's "What This Project Will NEVER
Bundle").

## Module layout

- `src/bs_roformer/bs_roformer.py`, `attend.py` -- the BS-RoFormer model
  architecture (from lucidrains/BS-RoFormer), with upstream-compatible
  `mlp_expansion_factor` pass-through for MaskEstimator checkpoint parity and
  explicit `mask_estimator_variant` selection for supported architecture heads.
- `src/bs_roformer/hyperace.py` -- the HyperACE mask-estimator variation used by
  pcunwa HyperACE v2 checkpoints. The RoFormer trunk remains in `bs_roformer.py`;
  this module owns only the segmentation head and its helper blocks.
- `src/bs_roformer/fno.py` -- the FNO mask-estimator variation used by
  pcunwa's instrumental FNO checkpoint. It reimplements the minimal FNO1d
  inference surface needed by the checkpoint instead of depending on the full
  `neuraloperator` research framework.
- `src/bs_roformer/large_inst.py` -- the Large-Inst mask-estimator variation
  used by pcunwa's `bs_large_v2_inst.ckpt`. The RoFormer trunk remains in
  `bs_roformer.py`; this module owns the four extra time/frequency Transformer
  pairs inserted before the mask MLP.
- `src/bs_roformer/model_registry.py` -- `BSModel` + `MODEL_REGISTRY`,
  backed by `config/checkpoints.toml` so new models don't need a code change.
  `MODEL_REGISTRY.get()` accepts slug, friendly name, or checkpoint filename.
- `src/bs_roformer/download.py` -- checkpoint/config download, sha256
  verification, models-dir resolution, and the `bs-roformer-download` CLI.
  `ensure_model_assets()` is the auto-download entry point `inference.py`
  calls on first use. Resolution precedence per asset: `config/checkpoints.toml`
  URL > packaged local file under `configs/` (configs only) >
  `DEFAULT_CKPT_BASE_URL`/`DEFAULT_CONFIG_BASE_URL` construction (legacy
  fallback path only; the old TRvlvr repo is dead -- see "Weights hosting"
  below).
- `src/bs_roformer/inference.py` -- the `bs-roformer-infer` CLI: folder-batch
  separation, chunked overlap-add, weights auto-resolve via `download.py`.
- `src/bs_roformer/utils.py` -- `demix_track`, `get_model_from_config`
  (converts YAML `!!python/tuple` lists back to real tuples post-safe-load).
- `src/bs_roformer/config/checkpoints.toml` -- the live registry source and
  patch point for dead URLs, with recorded sha256 + size per downloadable
  asset. Edit this file first before touching `download.py`.
- `src/bs_roformer/data/*.json` -- legacy compatibility/audit views kept in
  the repository but intentionally excluded from the wheel; runtime reads
  `config/checkpoints.toml`.
- `tools/check_weights_liveness.py` -- HEADs every registry URL; needs
  network, not run in default CI (see Testing below).

## Weights hosting (org constitution article 4)

All 23 registry models download from third-party hosts at runtime; none are
committed to this repo. Provenance has moved twice already, both discovered
by outage rather than announcement:

1. The original `jarredou` Hugging Face account behind the default
   BS-RoFormer-SW checkpoint was deleted (discovered 2026-06); repointed to
   `enerjazzer/BS-ROFO-SW-Fixed`.
2. The other 9 registry models fell back to the (now-dead) upstream
   `TRvlvr/model_repo` GitHub Releases URL. Re-audited and re-hosted
   2026-07-12 to `Politrees/UVR_resources` on Hugging Face, cross-verified
   sha256-identical against at least one other independent host per model
   before being written to the package registry -- full per-model provenance is
   in CHANGELOG.md's `[0.1.5]` entry. One exception: the De-Reverb model's
   *config* (not checkpoint) uses the author's
   `anvuew/dereverb_bs_roformer` copy rather than Politrees' similarly-named
   one, because the two diverge on `stft_hop_length` (512 vs 441) and the
   Politrees value silently degrades output rather than erroring -- do not
   "fix" this back to the Politrees config file.
3. The MVSep Mega 53-stem checkpoint comes from
   `ZFTurbo/Music-Source-Separation-Training` release `v1.0.21`; it strict-loads
   only when `mlp_expansion_factor: 2` reaches `MaskEstimator`. It is
   memory-heavy; upstream recommends at least 16GB VRAM and notes individual
   stems may be weaker than specialized models.
4. The 2026-07-23 BS-RoFormer scout added registry-only checkpoints that
   strict-loaded and passed a short forward probe: ZFTurbo MUSDB18HQ, anvuew
   vocals/MAG/karaoke/de-reverb, pcunwa Leap vocals/instrumental, and becruily
   karaoke. HyperACE v2 was then added as a supported MaskEstimator variation:
   registry metadata marks the two pcunwa HyperACE v2 checkpoints with
   `variation = "hyperace"`, which the CLI and clean API inject into model
   construction. The pcunwa FNO instrumental checkpoint was then added with
   `variation = "fno"` after strict-loading against the bundled minimal FNO1d
   implementation and passing a short forward probe.
5. pcunwa's Large-Inst instrumental checkpoint uses the normal trunk but adds
   four alternating time/frequency Transformer pairs inside the MaskEstimator.
   Registry metadata marks it with `variation = "large_inst"`; strict-loading
   `bs_large_v2_inst.ckpt` reports zero missing and zero unexpected keys, and a
   short forward probe returns finite output.

`config/checkpoints.toml` is the single patch point for a future re-host; it
does not require a code change. See README's "What This Project Will NEVER
Bundle" for the user-facing contract (auto-download, manual path, sha256
verification, cache location).

## Testing

Run with `uv run pytest -q` (installs via `uv sync --extra dev`; see
Development below). Test files:

- `tests/test_model_configs.py` -- every bundled/downloaded config loads via
  `yaml.safe_load()`, the `!!python/tuple`-to-tuple conversion runs, and
  models instantiate without beartype errors. Also runnable standalone
  (`python tests/test_model_configs.py`) for a human-readable summary.
- `tests/test_download.py` -- packaged-config matching and override-URL
  resolution regressions (covers two real release-blocking bugs: a filename
  mismatch that silently forced network fetches, and the jarredou outage --
  a future silent revert of the registry to a dead URL should fail this
  test, not ship quietly).
- `tests/test_weights_ux.py` -- sha256 verification wiring (a wrong hash
  must delete the file, not ship it) and models-dir resolution precedence,
  offline (temp files + monkeypatched `requests`).
- `tests/test_weights_liveness.py` -- HEADs every registry URL for real.
  Marked `network` and **deselected by default**
  (`addopts = "-m 'not network'"` in pyproject.toml); CI never needs network
  access. Run explicitly before a release: `pytest -m network
  tests/test_weights_liveness.py -v`, or `python
  tools/check_weights_liveness.py` directly.

CI (`.github/workflows/test.yml`) matrixes Python 3.10-3.13, all
`not network`-marked, all locally green as of 2026-07-12 (27 passed, 19
deselected on every version).

## File-top header convention

Every load-bearing module under `src/bs_roformer/` starts with a header of
this shape (as the module docstring): one-line title, then 2-3 sentences on
what the file is for and why it's shaped this way (including known failure
modes and the fix, where relevant), then a `Reads:` line naming what it
imports from inside the package. This convention is already in place across
the package (see `download.py`, `inference.py`, `model_registry.py`); keep
headers in sync as files change.

## Development

```bash
uv sync --extra dev      # install package + dev deps (pytest, ruff)
uv run pytest -q         # unit tests (network-marked tests deselected)
uv run ruff check .      # lint
```

`pip install -e ".[dev]"` is the pip-only equivalent (used by
`.github/workflows/publish.yml`'s release-gate test run).
## OpenMIRLab inference contract

This package is inference-only. Its clean facade provides an explicit lifecycle session (load, ready-only infer, release, close, status, cache_info, and context-manager support) while retaining legacy one-shot entry points for compatibility. Package-owned checkpoint configuration records URLs and integrity metadata; generic checkpoint overrides do not require code changes.
