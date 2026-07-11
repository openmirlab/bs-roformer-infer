"""pytest wrapper around tools/check_weights_liveness.py -- network-marked, skipped by default.

Deselected by the project's default `addopts = "-m 'not network'"` (see pyproject.toml)
so CI and local `pytest tests/` runs never need network access. Run explicitly with
`pytest -m network tests/test_weights_liveness.py -v` to actually HEAD every
registry URL, e.g. before a release, to catch a dead mirror before users do.

Reads: tools.check_weights_liveness, bs_roformer.model_registry.MODEL_REGISTRY
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.check_weights_liveness import check_url, iter_registry_urls

pytestmark = pytest.mark.network


@pytest.mark.parametrize("label,url", list(iter_registry_urls()))
def test_registry_url_is_live(label: str, url: str):
    ok, detail = check_url(url)
    assert ok, f"{label} is not reachable: {url} ({detail})"
