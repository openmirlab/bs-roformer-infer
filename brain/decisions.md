# Decisions

Settled calls and their reasoning. A decision here is binding on
[`plan.md`](plan.md); reversing one means editing this file, not quietly
diverging in code.

## Settled

### D1 · `backend` and `device` are separate axes

`backend` selects the framework; `device` keeps its existing Torch meaning. This
is not a new invention — it is the contract the org already wrote for the
`ddc-onset-infer` MLX pilot, reused verbatim so two packages do not grow two
vocabularies for one idea.

### D2 · Every addition is additive

No existing import, signature, CLI flag, default, or output may change. A caller
who never mentions `backend` gets exactly today's behaviour on today's devices.
The one deliberate exception is D8.

### D3 · Vendor the MLX model, do not depend on `mlx-audio-separator`

Taking the published package would drag in its model catalog, CLI, Demucs/VR/MDX
architectures, and `mlx-audio-io`. Article 4a says each package owns its own
loading and inference; vendoring ~1 400 lines of MIT-licensed roformer model code
satisfies that, a whole-toolkit dependency does not.

Vendored code carries upstream attribution verified against primary sources
(article 5), and its source revision is recorded.

### D4 · Depend on `mlx-spectro`; vendoring is the documented fallback

It is MIT, pure Python, declares a version floor rather than a ceiling, and
depends only on `mlx` + `numpy`. At 6 072 lines it is a real library, not a shim,
so re-implementing spectral ops would be gratuitous.

Recorded risk: `mlx-spectro` and the vendored model code share one individual
maintainer, and "the upstream went quiet" is the exact failure this organization
exists to absorb. Mitigation is not avoidance — it is that the package is small
enough and permissive enough to vendor on the day it stops moving. Written down
so the fallback is a plan rather than a scramble.

### D5 · Block `mlx-audio-io`

The roformer path never imports it; `soundfile` already covers I/O; it ships a
compiled AudioToolbox backend (article 3 forbids compiled dependencies in a core
install); and it pins `mlx==0.31.2` exactly, which article 3 forbids and which is
the same shape as the numba/llvmlite backtrack trap already recorded in the
constitution.

### D6 · `mlx` lives in an optional extra

`pip install bs-roformer-infer` must remain MLX-free and import-clean on every
platform. MLX arrives through `[mlx]`.

### D7 · MLX mixed precision defaults to off

Measured: no speed gain on this path, parity degraded ~87x. Article 2 requires
accuracy-affecting optimizations to be opt-in and default off. The upstream
loader does `os.environ.setdefault("MLX_ENABLE_AMP", "1")`; we set it explicitly
to `"0"` rather than inheriting that default.

### D8 · `device="mps"` stops raising — a deliberate contract reversal

`tests/test_lifecycle_contract.py:16` currently asserts that `"mps"` raises. That
negative contract is being reversed on measured evidence (12/12 operators native,
parity 6.6e-08). Article 8 requires this class of change to be called out
explicitly in the commit message and CHANGELOG rather than folded into a diff.

### D9 · MPS lands before MLX

Three reasons, in order: it unblocks all 24 registry models where MLX initially
covers 20; the MLX pilot contract itself requires a Torch/MPS baseline to
benchmark against, which the package cannot currently produce; and it is the
smaller, lower-risk change, so it de-risks the shared plumbing (backend
selection, session lifecycle, cache release) before MLX rides on it.

### D10 · Parity tolerance is ours, not upstream's

`mlx-audio-separator` accepts `rel_L2 <= 5e-2`. Measured reality is 7.27e-07.
Gates are set from what the implementation actually achieves, and a tolerance
breach is investigated rather than widened (article 2).

### D11 · Ship 24/24 — all three variant heads get an MLX port

Paul, 2026-07-30. Not a capability gap: every primitive the three heads need was
verified present in MLX. They are absent because they are this project's own
additions to the Torch side, which upstream had no reason to implement.

### D12 · `backend="auto"` prefers MLX where it can actually run

Paul, 2026-07-30. `auto` tries MLX first and falls back to Torch.

Corollary found while implementing, and fixed: `auto` must know the checkpoint's
registry variation *before* it settles, or it picks MLX for a checkpoint MLX has no
head for and then fails at construction, where Torch would simply have worked.
Resolution now takes `variation`. An **explicit** backend is still never
downgraded — that request is honoured or refused.

### D13 · Share the *pattern*, not a code library — decided at N=2

Porting `melband-roformer-infer` was the second consumer, and the constitution's
N+1 rule says that is when you learn what actually repeats. It repeated less than
expected, and the parts that did not repeat are the dangerous ones.

**Transferred perfectly** (the checklist): the `backend` × `device` contract, the
seam at a whole mixture, `exact_zero_safe_rfft`, the auditing weight loader,
"your parity fixture must contain silence", and "validate the test by removing
the fix".

**Did not transfer, and would have shipped a wrong model if it had:**

- `melband`'s Torch `MLP()` builds `depth` hidden layers; this package's builds
  `depth - 1`. Reusing this package's MLX primitive verbatim would have produced a
  shallower MLP than any melband checkpoint was trained with.
- Upstream's vendored MLX melband applies a trunk-level `final_norm` that
  melband's own Torch model never trains.

Both are genuine architectural divergence between two sibling packages, not
copy-paste error. A shared code library for the model math would have to
parameterize exactly the things that differ, and would have silently
re-introduced the first bug unless it forced re-verification against each
package's own Torch model anyway.

**So: what kept the port correct was the auditing loader, not shared code.** A
shared primitives repo stays worth having eventually — the Metal rfft workaround
and the layout remaps are genuinely identical everywhere — but it is a small
box of stateless helpers, and it must never grow into the shared runtime
article 4a forbids. Revisit at N=3 or N=4, with this evidence in hand.

## Open

| # | Question | Owner | Blocks |
|---|---|---|---|
| O2 | How the `ddc-onset-mlx-pilot` Non-goal ("do not add MLX support to another package during this pilot") gets amended | Paul | MLX merge, not MLX work |

O2 does not block implementation or evidence-gathering; it blocks landing MLX on
mainline. Recording it here so it is a scheduled decision rather than a surprise.
