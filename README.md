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
- **Auto-Download**: Automatic checkpoint downloads with hash verification
- **10 Pre-trained Models**: Vocals, instrumentals, and multi-stem separation
- **CLI Tools**: `bs-roformer-infer` and `bs-roformer-download` commands
- **Python API**: Clean programmatic interface
- **Model Registry**: Easy model discovery and management

---

## Quick Start

### Installation

```bash
# Using pip
pip install bs-roformer-infer

# Using UV (recommended)
uv pip install bs-roformer-infer
```

### Download Models

```bash
# List available models
bs-roformer-download --list-models

# Download the recommended model (BS-RoFormer-SW)
bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou

# Download all models
bs-roformer-download --all --output-dir ./models
```

### CLI Inference

```bash
# Using the recommended BS-RoFormer-SW model
bs-roformer-infer \
  --config_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.yaml \
  --model_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Rofo-SW-Fixed.ckpt \
  --input_folder ./songs \
  --store_dir ./outputs
```

### Python API

```python
from pathlib import Path
from ml_collections import ConfigDict
import torch
import yaml
from bs_roformer import MODEL_REGISTRY, DEFAULT_MODEL, get_model_from_config

# Use the default recommended model (BS-RoFormer-SW)
entry = MODEL_REGISTRY.get(DEFAULT_MODEL)

# Load config and model
config = ConfigDict(yaml.safe_load(open(f"models/{entry.slug}/{entry.config}")))
model = get_model_from_config("bs_roformer", config)
state_dict = torch.load(f"models/{entry.slug}/{entry.checkpoint}", map_location="cpu")
model.load_state_dict(state_dict)
```

---

## Recommended Model

**BS-RoFormer-SW** (`roformer-model-bs-roformer-sw-by-jarredou`) is the recommended default model for audio source separation. It supports **6-stem separation** (vocals, drums, bass, guitar, piano, other) and provides excellent quality for production workflows.

```python
from bs_roformer import DEFAULT_MODEL
print(DEFAULT_MODEL)  # "roformer-model-bs-roformer-sw-by-jarredou"
```

---

## Available Models

| Model | Category | Description |
|-------|----------|-------------|
| **`roformer-model-bs-roformer-sw-by-jarredou`** | 6-stem | **Recommended** - BS-RoFormer SW (vocals, drums, bass, guitar, piano, other) |
| `roformer-model-bs-roformer-vocals-resurrection-by-unwa` | vocals | Vocals Resurrection by unwa |
| `roformer-model-bs-roformer-vocals-revive-v3e-by-unwa` | vocals | Vocals Revive V3e by unwa |
| `roformer-model-bs-roformer-instrumental-resurrection-by-unwa` | instrumental | Instrumental Resurrection by unwa |
| `roformer-model-bs-roformer-de-reverb` | dereverb | De-reverberation model |
| ... | ... | See `--list-models` for full list |

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
results = MODEL_REGISTRY.search("viperx")
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
