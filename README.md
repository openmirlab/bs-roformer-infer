# BS-RoFormer-Infer

**Production-ready, inference-only toolkit for Band-Split RoPE Transformer audio source separation**

BS-RoFormer-Infer provides a clean, lightweight API for running music source separation inference using Band-Split RoFormer models with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/bs-roformer-infer)](https://pypi.org/project/bs-roformer-infer/)

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: the default model is fetched on first use and sha256-verified against recorded checksums
- **CLI Tools**: `bs-roformer-infer` and `bs-roformer-download` commands
- **Python API**: Clean programmatic interface
- **Model Registry**: Easy model discovery with search and category filtering

---

## Quick Start

### Installation

```bash
# Using pip
pip install bs-roformer-infer

# Using UV (recommended)
uv pip install bs-roformer-infer
```

### CLI Inference

```bash
# First run auto-downloads the recommended BS-RoFormer-SW model (~700 MB,
# sha256-verified) into ~/.cache/bs-roformer-infer/ -- no separate download step needed
bs-roformer-infer --input_folder ./songs --store_dir ./outputs
```

Every WAV inside `input_folder` produces separated stems (vocals, drums, bass, guitar, piano, other) plus `*_instrumental.wav`. Explicit `--config_path`/`--model_path` arguments still work and skip auto-resolution entirely.

### Python API

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

---

## Model Weights

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

### Download CLI (manual path)

```bash
# List available models
bs-roformer-download --list-models

# Download the recommended model into the cache dir
bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou

# Download into a custom directory
bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou --output-dir ./models
```

> **Note on download availability** (audited 2026-07-12): only the recommended
> BS-RoFormer-SW model currently has a live download source. The other 8
> registry entries fall back to the upstream TRvlvr repository, whose files
> have been removed (404 on both checkpoint and config) -- they cannot be
> downloaded until a live mirror is found. Run
> `python tools/check_weights_liveness.py` (needs network) to re-check.

---

## Recommended Model

**BS-RoFormer-SW** (`roformer-model-bs-roformer-sw-by-jarredou`) by jarredou is the recommended default model for audio source separation. It supports **6-stem separation** (vocals, drums, bass, guitar, piano, other) and provides excellent quality for production workflows.

```python
from bs_roformer import DEFAULT_MODEL
print(DEFAULT_MODEL)  # "roformer-model-bs-roformer-sw-by-jarredou"
```

---

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

> As of the 2026-07-12 liveness audit, only the recommended
> `roformer-model-bs-roformer-sw-by-jarredou` model has a live download URL;
> the remaining registry entries are currently unavailable upstream (see the
> availability note in [Model Weights](#model-weights)).

---

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

## Development Installation

```bash
# Clone repository
git clone https://github.com/openmirlab/bs-roformer-infer.git
cd bs-roformer-infer

# Install with UV
uv sync

# Install with pip
pip install -e ".[dev]"
```

---

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)** by Phil Wang (lucidrains) - Clean PyTorch implementation of the Band-Split RoPE Transformer architecture
- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** by Andrew Beveridge (nomadkaraoke) - Pre-trained checkpoints and model configurations
- **Original Research** - Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung for the Band-Split RoPE Transformer paper

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code and configurations adapted from:
- **BS-RoFormer** (MIT) - Phil Wang
- **python-audio-separator** (MIT) - Andrew Beveridge

---

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

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/bs-roformer-infer/issues](https://github.com/openmirlab/bs-roformer-infer/issues)

---
