"""Model registry for BS-Roformer checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODEL_DATA_PATH = _PACKAGE_ROOT / "data" / "bs_models.json"


@dataclass(frozen=True)
class BSModel:
    slug: str
    name: str
    checkpoint: str
    config: str
    category: str

    @property
    def default_sources(self) -> List[str]:
        if self.category in {"instrumental"}:
            return ["instrumental", "vocals"]
        if self.category in {"karaoke", "vocals"}:
            return ["vocals", "other"]
        if self.category == "dereverb":
            return ["dry", "wet"]
        return ["vocals", "other"]


class ModelRegistry:
    def __init__(self):
        data = json.loads(_MODEL_DATA_PATH.read_text())
        self._models: Dict[str, BSModel] = {}
        for slug, meta in data["models"].items():
            model = BSModel(
                slug=slug,
                name=meta["name"],
                checkpoint=meta["checkpoint"],
                config=meta["config"],
                category=meta.get("category", "general"),
            )
            self._models[slug] = model
        # simple lookup by normalized name or checkpoint
        self._by_name = {model.name.lower(): model.slug for model in self._models.values()}
        self._by_checkpoint = {model.checkpoint.lower(): model.slug for model in self._models.values()}

    def list(self, category: Optional[str] = None) -> List[BSModel]:
        if category is None:
            return sorted(self._models.values(), key=lambda m: m.name)
        return sorted(
            (m for m in self._models.values() if m.category == category.lower()),
            key=lambda m: m.name,
        )

    def categories(self) -> List[str]:
        return sorted({m.category for m in self._models.values()})

    def get(self, key: str) -> BSModel:
        normalized = key.lower()
        if normalized in self._models:
            return self._models[normalized]
        if normalized in self._by_name:
            return self._models[self._by_name[normalized]]
        if normalized in self._by_checkpoint:
            return self._models[self._by_checkpoint[normalized]]
        raise KeyError(f"Unknown BS-Roformer model: {key}")

    def search(self, term: str) -> List[MelBandModel]:
        normalized = term.lower()
        return [
            m for m in self._models.values()
            if normalized in m.name.lower() or normalized in m.checkpoint.lower()
        ]

    def as_table(self, category: Optional[str] = None) -> str:
        rows = self.list(category)
        if not rows:
            return "No models registered."
        name_w = max(len(m.name) for m in rows)
        cat_w = max(len(m.category) for m in rows)
        lines = [f"{'Name'.ljust(name_w)}  {'Category'.ljust(cat_w)}  Checkpoint"]
        lines.append("-" * len(lines[0]))
        for model in rows:
            lines.append(
                f"{model.name.ljust(name_w)}  {model.category.ljust(cat_w)}  {model.checkpoint}"
            )
        return "\n".join(lines)


MODEL_REGISTRY = ModelRegistry()

__all__ = ["BSModel", "ModelRegistry", "MODEL_REGISTRY"]
