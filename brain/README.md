# brain — multi-backend BS-RoFormer

Working memory for one campaign: give `bs-roformer-infer` an **MLX backend** and
an **MPS device**, so a caller switches hardware paths by changing one argument
and loses nothing.

This directory is maintainer-facing scratch canon. It is not the package
contract — `README.md` (users) and `CLAUDE.md` (maintainers) stay authoritative.
When a decision here ships, restate it there and note it here as landed.

## Navigate

| File | What it answers |
|---|---|
| [`evidence.md`](evidence.md) | What was actually measured on real hardware, and what remains unmeasured |
| [`decisions.md`](decisions.md) | What is settled, why, and what is still open |
| [`architecture.md`](architecture.md) | The `backend` × `device` contract and the public API shape |
| [`plan.md`](plan.md) | Phased implementation with its acceptance gates |

## The one-line problem

`device` and `backend` are two different axes and the package currently only has
a partial version of the first:

- **`device`** — which chip Torch computes on. Today: `cpu`, `cuda`, `cuda:N`.
  Missing: `mps`, even though every load-bearing operator was measured to run on
  it natively.
- **`backend`** — which framework computes at all. Today: only Torch. MLX runs
  the same checkpoint ~2.2x faster than Torch on MPS, at float32 noise parity.

Both are additive. Neither may remove or degrade an existing path.
