# Architecture — the `backend` × `device` contract

## The two axes

```
backend = "torch" (default) | "mlx" | "auto"      which framework computes
device  = None | "auto" | "cpu" | "cuda" | "cuda:N" | "mps"    where Torch computes
```

`backend` chooses the implementation. `device` keeps its existing Torch meaning
and is not overloaded to mean "Apple Silicon". The MLX backend owns its own
execution target and rejects a Torch device string it cannot honour rather than
quietly reinterpreting it (article 4b: an explicit choice is honoured or the call
fails loudly).

### Resolution table

| `backend` | Resolves to | `device` accepted |
|---|---|---|
| `"torch"` (default) | Torch | `None`, `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"` |
| `"mlx"` | MLX, or raise if unavailable | `None`, `"auto"`, `"mps"` — anything else raises |
| `"auto"` | MLX on Apple Silicon when the `[mlx]` extra is importable, else Torch | per resolved backend |

`backend` defaults to `"torch"`, not `"auto"`, so an existing caller's behaviour
is bit-for-bit untouched. `"auto"` is an opt-in convenience.

### `device="auto"` keeps its legacy meaning

`auto` resolves to `cuda:0` if CUDA is available, else `cpu` — **not** MPS.
Sibling packages in the org's device campaign preserved whatever their legacy
auto-selection did, and this package's legacy auto has never chosen MPS. Silently
promoting Mac users onto a different compute path would change their outputs
(measured 6.6e-08, small but not zero) and their performance profile without
their asking.

Mac users opt in with `device="mps"`. Whether a future major flips this default
is recorded as an open question, not decided here.

## Where the seam goes

The narrow interface is **one whole separation**, not one forward pass:

```python
class SeparationBackend(Protocol):
    name: str

    @classmethod
    def is_available(cls) -> bool: ...

    @property
    def resolved_device(self) -> str: ...

    def separate(self, mix: np.ndarray) -> dict[str, np.ndarray]: ...
    def release(self) -> None: ...
```

`mix` is `(channels, samples)` float32; the return maps stem name to
`(channels, samples)`. NumPy is the boundary currency because it is the one array
type both frameworks already convert to, and because it is what the file-writing
layer above already consumes.

**Why the seam sits this high.** Chunking, windowing, and overlap-add accumulate
on-device for speed — Torch accumulates in `torch.zeros(...).to(device)`, MLX in
`mx.array`. A lower seam (per-chunk forward, arrays crossing every chunk) would
force those accumulators back to the host on every step and hand the performance
win straight back. So each backend owns its whole chunked inference, and
everything above the seam is written once:

| Layer | Owner |
|---|---|
| CLI parsing, folder iteration, manifest | shared (`inference.py`) |
| session lifecycle, `status`, `cache_info()` | shared (`clean_api.py`) |
| checkpoint download, sha256, models-dir resolution | shared (`download.py`) |
| config loading, `!!python/tuple` handling | shared (`inference.py`) |
| instrumental derivation, file writing | shared (`inference.py`) |
| **model construction, checkpoint→weights, chunking, device** | **backend** |

This is the information-hiding line: nothing above the seam knows a tensor
layout, a dtype, or which chip is busy.

## Module layout

```
src/bs_roformer/
  backends/
    __init__.py          barrel + resolve() — the only thing callers touch
    base.py              SeparationBackend protocol, shared validation
    torch_backend.py     wraps today's get_model_from_config + demix_track
    mlx_backend.py       MLX runtime; every mlx import is lazy
  mlx/                   vendored MLX model (only imported by mlx_backend)
    model.py             vendored BSRoformerMLX (MIT, revision recorded)
    convert.py           torch state_dict -> MLX weights, with a match audit
    heads/               ported hyperace / fno / large_inst
```

`import bs_roformer` must not import `mlx`. `backends/__init__.py` resolves by
name and only touches `mlx_backend` when asked for it.

## Public surface after the change

Additive only. Everything in today's `__all__` keeps its name, signature, and
behaviour.

```python
# new keyword, everywhere a device is already accepted
BSRoformerSession(model_name=..., backend="mlx", device="auto")
BSRoformerSeparator(backend="auto")
separate_folder(folder, backend="mlx")

# CLI
bs-roformer-infer --input_folder in --backend mlx
bs-roformer-infer --input_folder in --device mps
```

`demix_track` and `get_model_from_config` stay exported and stay Torch — they are
the advanced composition path article 4a forbids removing, and the Torch backend
calls the same functions rather than a private fork of them.

## Failure modes this design must not have

1. `backend="mlx"` on an Intel Mac or Linux box → clear error naming the extra,
   never a silent Torch fallback.
2. `device="cuda"` with `backend="mlx"` → raise; do not reinterpret.
3. A converted MLX checkpoint that silently leaves parameters at random init.
   `load_weights(strict=False)` upstream makes this possible, so conversion
   audits the match count and raises on any shortfall.
4. `cache_info()` reporting differently depending on backend — it reads the same
   resolver either way (article 4).
5. Two implementations that can disagree without anyone noticing — every
   supported checkpoint carries a Torch-vs-MLX parity fixture.
