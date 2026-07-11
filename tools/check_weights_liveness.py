#!/usr/bin/env python3
"""Weights-liveness check -- HEADs every registry download URL, fails on non-2xx/3xx.

Prototype for an org-wide "is the third-party host still there" gate: this package
shipped a real outage (the jarredou HF account behind BS-RoFormer-SW was deleted,
2026-06) that nothing caught until users reported broken downloads. This script
walks MODEL_REGISTRY, resolves each checkpoint/config URL through the same
override-then-packaged-then-default precedence download.py uses, and HEADs it.
Run manually (`python tools/check_weights_liveness.py`) or via
`pytest -m network tests/test_weights_liveness.py` -- neither runs in the default
test suite since both need real network access.

Reads: bs_roformer.model_registry.MODEL_REGISTRY, bs_roformer.download, requests
"""

from __future__ import annotations

import sys
from typing import Iterator, Tuple

import requests

from bs_roformer.download import _checkpoint_url, _config_url
from bs_roformer.model_registry import MODEL_REGISTRY

TIMEOUT_SECONDS = 15


def iter_registry_urls() -> Iterator[Tuple[str, str]]:
    """Yield (label, url) for every network-fetchable asset in the registry.

    Configs that resolve to a packaged local file (`_config_url` returns None)
    are skipped -- there is no URL to check, that's the point of packaging them.
    """
    for model in MODEL_REGISTRY.list():
        ckpt_url = _checkpoint_url(model)
        if ckpt_url:
            yield f"{model.slug} checkpoint ({model.checkpoint})", ckpt_url

        config_url = _config_url(model)
        if config_url:
            yield f"{model.slug} config ({model.config})", config_url


def check_url(url: str, timeout: float = TIMEOUT_SECONDS) -> Tuple[bool, str]:
    """HEAD a URL, falling back to a ranged GET if the host doesn't support HEAD."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code in (405, 501):
            resp = requests.get(
                url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=timeout
            )
        ok = 200 <= resp.status_code < 400
        return ok, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"error: {exc}"


def main() -> int:
    failures = []
    for label, url in iter_registry_urls():
        ok, detail = check_url(url)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}: {url} -> {detail}")
        if not ok:
            failures.append((label, url, detail))

    print()
    if failures:
        print(f"{len(failures)} URL(s) failed liveness check:")
        for label, url, detail in failures:
            print(f"  - {label}: {url} ({detail})")
        return 1

    print("All registry URLs are live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
