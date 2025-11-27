# bs-roformer-infer

Inference-only packaging of the Band-Split (BS) Roformer models used throughout
the worzpro stack. These checkpoints specialize in multi-stem source separation
and are compatible with the configs shipped in
`reference_modules/python-audio-separator`.

## Installation

```bash
cd patched_modules/bs-roformer-infer
uv sync
```

## Download models

```bash
uv run bs-roformer-download --list-models                     # inspect registry
uv run bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou
uv run bs-roformer-download --all --output-dir ./models
```

Each model is placed under `./models/<slug>/checkpoint+config`. The slug names
mirror `models.json`'s `roformer_download_list` entries for BS-Roformer
checkpoints so you can keep parity with audio-separator and Colab workflows.

## CLI inference

```bash
uv run bs-roformer-infer \
  --config_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Roformer-SW.yaml \
  --model_path models/roformer-model-bs-roformer-sw-by-jarredou/BS-Roformer-SW.ckpt \
  --input_folder ./songs \
  --store_dir ./outputs
```

## Python API

```python
from pathlib import Path
from ml_collections import ConfigDict
import torch
import yaml
from bs_roformer import MODEL_REGISTRY, get_model_from_config

entry = MODEL_REGISTRY.get("roformer-model-bs-roformer-sw-by-jarredou")
config = ConfigDict(yaml.safe_load(open(f"models/{entry.slug}/{entry.config}")))
model = get_model_from_config("bs_roformer", config)
state_dict = torch.load(f"models/{entry.slug}/{entry.checkpoint}", map_location="cpu")
model.load_state_dict(state_dict)
```

## Registry helpers

```python
from bs_roformer import MODEL_REGISTRY

print(MODEL_REGISTRY.categories())
for model in MODEL_REGISTRY.list("vocals"):
    print(model.name, model.checkpoint)
```

## License

[MIT](LICENSE)
