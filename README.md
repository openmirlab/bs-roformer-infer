# BS-RoFormer-Infer

**Production-ready, inference-only toolkit for Band-Split RoPE Transformer audio source separation**

BS-RoFormer-Infer provides a clean, lightweight API for running music source separation inference using Band-Split RoFormer models with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bs-roformer-infer)](https://pypi.org/project/bs-roformer-infer/)

---

## Why This Exists

BS-RoFormer (Band-Split RoPE Transformer) is a strong architecture for music
source separation, introduced by Lu, Wang, Kong, and Hung (2023). The
reference implementation, [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer),
provides the model architecture only -- no checkpoint management, no CLI, no
packaging for downstream use. Trained checkpoints are typically distributed
through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(which pulls in the full Ultimate Vocal Remover GUI stack) or through
individual community members' personal Hugging Face/Google Drive accounts --
hosts that can and do vanish without warning. This project's own history
includes exactly that: the original `jarredou` Hugging Face account behind
the default BS-RoFormer-SW checkpoint was deleted (discovered 2026-06), and
9 of the other 10 registry models' fallback URLs (the dead upstream
`TRvlvr/model_repo` GitHub repo) were found 404ing in a 2026-07-12 audit.

BS-RoFormer-Infer reprovides the architecture as a clean, pip-installable,
inference-only package: no training code, no GUI dependency, a versioned
model registry that can be repointed at a new host by editing one JSON file
(no code change), and sha256-verified auto-download so a corrupted or
tampered checkpoint is never silently loaded.

---

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)** by Phil Wang (lucidrains) -- Clean PyTorch implementation of the Band-Split RoPE Transformer architecture
- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** by Andrew Beveridge (nomadkaraoke) -- Pre-trained checkpoints and model configurations
- **Original Research** -- Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung for the Band-Split RoPE Transformer paper
- **[Politrees/UVR_resources](https://huggingface.co/Politrees/UVR_resources)** on Hugging Face -- current mirror host for 9 of the 10 registry checkpoints, after the original TRvlvr source went dead (see [Model Weights](#what-this-project-will-never-bundle))
- **[anvuew/dereverb_bs_roformer](https://huggingface.co/anvuew/dereverb_bs_roformer)** on Hugging Face -- author-hosted config for the De-Reverb model
- **[enerjazzer/BS-ROFO-SW-Fixed](https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed)** on Hugging Face -- current host for the default BS-RoFormer-SW checkpoint, after the original jarredou account was deleted

## Citation

If you use BS-RoFormer-Infer in your research, please cite the original paper:

```bibtex
@inproceedings{Lu2023MusicSS,
    title   = {Music Source Separation with Band-Split RoPE Transformer},
    author  = {Wei-Tsung Lu and Ju-Chiang Wang and Qiuqiang Kong and Yun-Ning Hung},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:261556702}
}
```

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: the default model is fetched on first use and sha256-verified against recorded checksums
- **CLI Tools**: `bs-roformer-infer` and `bs-roformer-download` commands
- **Python API**: Clean programmatic interface
- **Model Registry**: Easy model discovery with search and category filtering

## Scope

**In scope**: inference (forward pass) with the BS-RoFormer architecture; a
10-model registry spanning multi-stem, vocals, instrumental, and de-reverb
checkpoints; automatic, manual, and configurable-directory checkpoint
management with sha256 verification; a standalone download CLI.

**Out of scope, forever**:
- Training or fine-tuning code -- this package only ever runs a forward pass.
- The Ultimate Vocal Remover GUI itself, or any GUI.
- Bundling or committing checkpoint bytes to this repository's git history
  (see [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle)).

---

## Install

```bash
# Using pip
pip install bs-roformer-infer

# Using UV (recommended)
uv pip install bs-roformer-infer
```

## Quick Start

```bash
# First run auto-downloads the recommended BS-RoFormer-SW model (~700 MB,
# sha256-verified) into ~/.cache/bs-roformer-infer/ -- no separate download step needed
bs-roformer-infer --input_folder ./songs --store_dir ./outputs
```

Every WAV inside `input_folder` produces separated stems (vocals, drums, bass, guitar, piano, other) plus `*_instrumental.wav`. Explicit `--config_path`/`--model_path` arguments still work and skip auto-resolution entirely.

---

## Python API

```python
from ml_collections import ConfigDict
import torch
import yaml
from bs_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config
from bs_roformer.inference import SafeLoaderWithTuple

# Resolves local copies, or downloads (sha256-verified) on first use
ckpt_path, config_path = ensure_model_assets(DEFAULT_MODEL)

with open(config_path) as f:
    config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
model = get_model_from_config("bs_roformer", config)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
```

## Recommended Model

**BS-RoFormer-SW** (`roformer-model-bs-roformer-sw-by-jarredou`) by jarredou is the recommended default model for audio source separation. It supports **6-stem separation** (vocals, drums, bass, guitar, piano, other) and provides excellent quality for production workflows.

```python
from bs_roformer import DEFAULT_MODEL
print(DEFAULT_MODEL)  # "roformer-model-bs-roformer-sw-by-jarredou"
```

## Available Models

| Model | Category | Description |
|-------|----------|-------------|
| **`roformer-model-bs-roformer-sw-by-jarredou`** | multi-stem | **Recommended** - 6-stem separation (vocals, drums, bass, guitar, piano, other) |
| `roformer-model-bs-roformer-vocals-resurrection-by-unwa` | vocals | Vocals Resurrection by unwa |
| `roformer-model-bs-roformer-vocals-revive-v3e-by-unwa` | vocals | Vocals Revive V3e by unwa |
| `roformer-model-bs-roformer-vocals-revive-v2-by-unwa` | vocals | Vocals Revive V2 by unwa |
| `roformer-model-bs-roformer-vocals-revive-by-unwa` | vocals | Vocals Revive by unwa |
| `roformer-model-bs-roformer-vocals-by-gabox` | vocals | Vocals by Gabox |
| `roformer-model-bs-roformer-instrumental-resurrection-by-unwa` | instrumental | Instrumental Resurrection by unwa |
| `roformer-model-bs-roformer-de-reverb` | dereverb | De-reverberation model |
| ... | ... | See `--list-models` for full list |

**Categories**: multi-stem, vocals, instrumental, dereverb

> As of the 2026-07-12 re-audit, all registry entries have a live download
> URL (see the availability note in [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle)).

## Registry Helpers

```python
from bs_roformer import MODEL_REGISTRY

# List all categories
print(MODEL_REGISTRY.categories())

# List models by category
for model in MODEL_REGISTRY.list("vocals"):
    print(model.name, model.checkpoint)

# Search models
results = MODEL_REGISTRY.search("unwa")
for m in results:
    print(m.slug)

# Pretty-print all models
print(MODEL_REGISTRY.as_table())
```

---

## What This Project Will NEVER Bundle

Model weights are **never bundled or committed to this repository**. Every
checkpoint is downloaded at runtime from its registry-recorded source,
sha256-verified against `src/bs_roformer/data/checksums.json`, and cached
locally -- a mismatch deletes the file and retries instead of silently
keeping a corrupt checkpoint.

### Where weights live

Downloads default to `~/.cache/bs-roformer-infer/<model-slug>/`. The location is
configurable, resolved in this order:

1. Explicit argument: `--models_dir` (inference CLI), `--output-dir` (download CLI), or `ensure_model_assets(..., models_dir=...)` (API)
2. The `BS_ROFORMER_MODELS_PATH` environment variable
3. The default `~/.cache/bs-roformer-infer/`

A relative `./models` directory (the pre-0.1.4 default) is still searched as a
read fallback, so existing downloads keep working without re-fetching.

### Auto-download

When `bs-roformer-infer` runs without `--model_path`/`--config_path`, the
requested registry model (default: BS-RoFormer-SW) is looked up in the
directories above and downloaded on first use. Downloads are verified against
the sha256 checksums recorded in `src/bs_roformer/data/checksums.json`; a
mismatch deletes the file and retries instead of keeping a corrupt checkpoint.

### Manual download (offline / air-gapped)

The recommended BS-RoFormer-SW model needs one file (its config ships inside
the package):

| File | URL | sha256 |
|------|-----|--------|
| `BS-Rofo-SW-Fixed.ckpt` (699,412,152 bytes) | <https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt> | `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e` |

Place it at
`~/.cache/bs-roformer-infer/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt`
(or the equivalent path under your `BS_ROFORMER_MODELS_PATH`), and inference
will pick it up without network access.

The other 9 registry models (re-hosted 2026-07-12, see CHANGELOG for full
provenance) download from these mirrors:

| Model | File | URL | sha256 |
|-------|------|-----|--------|
| De-Reverb | `deverb_bs_roformer_8_384dim_10depth.ckpt` (361,499,604 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/deverb_bs_roformer_8_384dim_10depth.ckpt> | `9c38653aaa5e49f2f7b84dd3be2b6b679e0cbea23978e6b48389ee6f0a914768` |
| De-Reverb (config) | `deverb_bs_roformer_8_384dim_10depth_config.yaml` (2,358 bytes) | <https://huggingface.co/anvuew/dereverb_bs_roformer/resolve/main/archive/deverb_bs_roformer_8_384dim_10depth.yaml> **(author's file — NOT Politrees' similarly-named copy, which silently uses the wrong `stft_hop_length`; see CHANGELOG)** | `a87cf93b36b9a20d25a9cc4f78a2541ea0033988e7b6c38dcf0029e9290af816` |
| Chorus Male-Female by Sucial | `model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt` (527,121,477 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt> | `123c00786bdbc6bd462dddb35cd21fd6ae99ab8319f93f63a8abc1012e593d94` |
| Instrumental Resurrection by unwa | `bs_roformer_instrumental_resurrection_unwa.ckpt` (204,483,033 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/model_BandSplit-Roformer_Resurrection_Instrumental_by-Unwa.ckpt> | `16311025a5133ae6411760ccfe9e3e66b31a01d9d8bec0a03fa7ec4bedac7a15` |
| Male-Female by aufr33 | `bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt` (527,119,779 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt> | `3cf11736d1b42a11ae55d8299316585921477dd2a671b24b663660846ca9861b` |
| Vocals by Gabox | `bs_roformer_vocals_gabox.ckpt` (639,254,584 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/bs_roformer_voc_gabox.ckpt> | `18d58efe5e949e70fab11b875329af6d06ef11ccc29574bfe943fb57cc827f38` |
| Vocals Resurrection by unwa | `bs_roformer_vocals_resurrection_unwa.ckpt` (204,510,749 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/model_BandSplit-Roformer_Resurrection_Vocals_by-Unwa.ckpt> | `9dbfe5cb572e4ed32a15ec727d7bd06c8d7aba97509e6fda5bc008bb1e0b2dd5` |
| Vocals Revive by unwa | `bs_roformer_vocals_revive_unwa.ckpt` (639,326,600 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/bs_roformer_revive_by_unwa.ckpt> | `f1d7e4bfdfef07c6b2bc1d65283a7d03c3c38f8c7dbc8d729b785f93c8b8699a` |
| Vocals Revive V2 by unwa | `bs_roformer_vocals_revive_v2_unwa.ckpt` (639,326,600 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/bs_roformer_revive_v2_by_unwa.ckpt> | `58098850c882a7472dad39f99fb8040ce6eaafe671cfe9881d89aea276bbb5f5` |
| Vocals Revive V3e by unwa | `bs_roformer_vocals_revive_v3e_unwa.ckpt` (639,326,600 bytes) | <https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/bs_roformer_revive_v3_by_unwa.ckpt> (hosted there without the trailing "e" — same file) | `1b0751b9a15c591407c3b77f08eb4ad3005e42e96051f3f2b39760f1130c467b` |

The Chorus/Male-Female-aufr33 config
(`config_chorus_male_female_bs_roformer.yaml`) and the three Revive
checkpoints' shared config (`config_bs_roformer_vocals_revive_unwa.yaml`) are
also fetched from Politrees/UVR_resources — see `data/overrides.json` for the
exact URLs and `data/checksums.json` for their hashes.

### Download CLI (manual path)

```bash
# List available models
bs-roformer-download --list-models

# Download the recommended model into the cache dir
bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou

# Download into a custom directory
bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou --output-dir ./models
```

> **Note on download availability** (re-audited 2026-07-12): all 10 registry
> models now have live download sources. The 9 that fell back to the dead
> upstream TRvlvr repository were re-hosted to Politrees/UVR_resources (with
> the De-Reverb config sourced from the author's anvuew/dereverb_bs_roformer
> repo instead — see CHANGELOG for why). Run
> `python tools/check_weights_liveness.py` (needs network) to re-check.

---

## Development

```bash
# Clone repository
git clone https://github.com/openmirlab/bs-roformer-infer.git
cd bs-roformer-infer

# Install with UV
uv sync --extra dev

# Install with pip
pip install -e ".[dev]"
```

```bash
uv run pytest -q       # unit tests (network-marked tests deselected by default)
uv run ruff check .    # lint
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code and configurations adapted from:
- **BS-RoFormer** (MIT) - Phil Wang
- **python-audio-separator** (MIT) - Andrew Beveridge

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/bs-roformer-infer/issues](https://github.com/openmirlab/bs-roformer-infer/issues)

---
