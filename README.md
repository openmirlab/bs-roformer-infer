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

# Download a specific model
bs-roformer-download --model bs-roformer-viperx-1297

# Download all models
bs-roformer-download --all --output-dir ./models
```

### CLI Inference

```bash
bs-roformer-infer \
  --config_path models/bs-roformer-viperx-1297/model_bs_roformer_ep_317_sdr_12.9755.yaml \
  --model_path models/bs-roformer-viperx-1297/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --input_folder ./songs \
  --store_dir ./outputs
```

### Python API

```python
from pathlib import Path
from ml_collections import ConfigDict
import torch
import yaml
from bs_roformer import MODEL_REGISTRY, get_model_from_config

# Get model info from registry
entry = MODEL_REGISTRY.get("bs-roformer-viperx-1297")

# Load config and model
config = ConfigDict(yaml.safe_load(open(f"models/{entry.slug}/{entry.config}")))
model = get_model_from_config("bs_roformer", config)
state_dict = torch.load(f"models/{entry.slug}/{entry.checkpoint}", map_location="cpu")
model.load_state_dict(state_dict)
```

---

## Available Models

| Model | Category | Description |
|-------|----------|-------------|
| `bs-roformer-viperx-1297` | vocals | ViperX vocals model (SDR 12.97) |
| `bs-roformer-viperx-1053` | vocals | ViperX vocals model (SDR 10.53) |
| `mel-roformer-viperx-1143` | vocals | MelBand RoFormer vocals |
| `bs-roformer-de-reverb` | other | De-reverberation model |
| `roformer-model-bs-roformer-sw-by-jarredou` | vocals | BS-RoFormer SW by Jarredou |
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
