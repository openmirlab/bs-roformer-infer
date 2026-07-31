# Plan

Five phases. Each has an acceptance gate; a phase is not done until its gate is
evidenced, and phases land as separate commits (article 4a keeps structural
refactor separate from behaviour change).

Status legend: `todo` · `doing` · `done`

---

## Phase 1 · MPS on the Torch backend — `done` (2026-07-30)

Landed. Default suite `46 passed, 48 deselected` (baseline was 43 passed);
`pytest -m realweights tests/test_device_parity.py` passes in ~2 min; `ruff`
error count on the touched files is unchanged from HEAD. Docs updated in the same
window: `README.md`, `CLAUDE.md`, `CHANGELOG.md`, and the org's
`DEVICE_AND_MODEL_CACHE.md` row plus a change-log entry carrying the two
org-wide findings (x86_64 `uv` trap; packaged-config vs registry-metadata drift).

<details><summary>original phase spec</summary>


The `device` axis only. No new modules, no MLX, no seam.

**Changes**

| File | Change |
|---|---|
| `inference.py:270` `_select_device` | accept `"mps"`; explicit-but-unavailable raises (never silently downgrade); `auto` keeps its legacy `cuda-else-cpu` meaning |
| `inference.py:11-12` header | stale — claims a CPU fallback that the code no longer does. Fix in the same diff |
| `inference.py:224` | `torch.backends.cudnn.benchmark = True` is unconditional; scope it to CUDA |
| `utils.py:94` | `torch.cuda.amp.autocast()` is deprecated and already a measured no-op off CUDA (it warns `CUDA is not available. Disabling autocast.`). Replace with an explicitly CUDA-scoped autocast so behaviour is identical and the warning stops |
| `attend.py:93` | `torch.backends.cuda.sdp_kernel` is deprecated and fires once per attention call. Move to `torch.nn.attention.sdpa_kernel` where available, keeping the same backend selection |
| `clean_api.py:129` | `release()` empties only the CUDA cache; add the MPS cache |
| `tests/test_lifecycle_contract.py:16` | reverse the negative contract (D8) — `"mps"` must now resolve, and a new test must cover explicit-unavailable-raises |

**Gate**

1. Full suite green (baseline before this phase: `43 passed, 47 deselected`).
2. New device tests: `mps` resolves when available; `mps` raises when not; `auto`
   still returns `cpu` on a CUDA-less box; `cuda:N` behaviour unchanged.
3. A recorded CPU-vs-MPS comparison on the real default checkpoint, with the
   tolerance written down and the environment (torch build, device) recorded
   alongside it per article 2.
4. `README.md`, `CLAUDE.md`, `CHANGELOG.md` updated in the same window, plus
   `openmirlab-dev/docs/DEVICE_AND_MODEL_CACHE.md` — whose row for this package
   currently claims CUDA-index-only selection.
5. The negative-contract reversal is named explicitly in the commit message and
   CHANGELOG (article 8), not folded in silently.

</details>

---

## Phase 2 · The backend seam — `done` (2026-07-30)

Landed. `backends/` holds the `SeparationBackend` protocol, `BackendUnavailable`,
a name resolver, and `TorchBackend` wrapping the shipped `demix_track` path.
`separate_folder_with()` owns everything backend-agnostic; `run_folder()` keeps
its exact signature and delegates. `backend=` is wired through session,
separator, `separate_folder()`, and `--backend`; `cache_info()` reports the
resolved backend and device.

**Gate met.** Suite `52 passed, 48 deselected`. Output verified **exactly
identical** (0.0 max abs across all seven outputs, including the derived
instrumental) against pre-refactor HEAD on the real default checkpoint.
End-to-end CLI verified: `--device mps` produces stems; `--backend mlx` raises
before any checkpoint work; `--backend auto` resolves to Torch and is
bit-identical to the default path.

**Two gate corrections, recorded rather than smoothed over:**

1. *"Zero test changes beyond new seam tests" was too strict and was relaxed.*
   Three tests patched `inference.demix_track` / `inference.run_folder`; those
   call sites legitimately moved with the seam, so their patch targets moved too.
   What each test proves is unchanged — device forwarding is now asserted through
   the public `cache_info()` instead of a patched internal.
2. *Byte-comparing WAV files is not a valid regression gate.* The first A/B
   reported all seven outputs differing. Exactly one byte differed, inside
   libsndfile's `PEAK` chunk, which carries a write timestamp — the same code run
   twice differs there too. Decoded samples were identical. **Any golden-fixture
   gate in this org that byte-compares audio files is measuring the clock.**

---

## Phase 3 · MLX backend — `done` (2026-07-30)

Vendored model in `src/bs_roformer/mlx/` (upstream revision recorded, attribution
verified against the LICENSE), auditing weight loader that raises rather than
loading partially, `[mlx]` extra, AMP off, variation and chunk-alignment refusals,
`backend=` wired through session, separator, `separate_folder()`, and CLI.

**The silence bug (§14) was root-caused and fixed**, not worked around blindly:
MLX's Metal `rfft` returns ~4.5e-07 for exactly-zero frames, `L2Norm` amplifies
that a millionfold because it discards magnitude, and time-axis attention spreads
it everywhere. Routing `rfft` through the CPU stream during the STFT removes it at
source, at no measurable speed cost.

**Gate met.** End to end through the public session API on the real checkpoint:
all seven outputs within `3.427e-07` of Torch, 9.3 s against Torch-MPS's 25 s.
`tests/test_mlx_parity.py` (3 cases) passes and was validated by removing the fix
and confirming the silent cases fail. Default suite `59 passed, 51 deselected`;
`import bs_roformer` remains MLX-free.

Outstanding for this phase: parity on **real music** rather than seeded noise, and
per-checkpoint conversion audits for the other 19 stock models (only the default is
verified at checkpoint level).

<details><summary>original phase spec</summary>


Vendor the MLX model (D3), depend on `mlx-spectro` (D4), block `mlx-audio-io`
(D5), `[mlx]` extra (D6), AMP off (D7).

**Sequence**

1. Vendor `BSRoformerMLX` with its upstream revision and attribution recorded;
   verify the attribution against primary sources before writing it (article 5).
2. Port `convert_torch_to_mlx_weights`, replacing `load_weights(strict=False)`
   with an auditing loader that raises on any parameter left at random init —
   upstream's silent-drop behaviour is a correctness hazard (architecture,
   failure mode 3).
3. **Reject variant checkpoints explicitly.** Evidence §9: all four
   `hyperace`/`fno`/`large_inst` models construct under MLX *without raising* and
   build the plain `MaskEstimator` instead. Silent wrong architecture is the
   worst available outcome, so `backend="mlx"` must refuse a checkpoint whose
   registry `variation` has no MLX head, naming the reason.
4. Fix the `freqs_per_bands` hard-index (dormant today — all 24 registry configs
   happen to set it — but the package's own *packaged* config does not, per §10).
5. Forward `stft_normalized` into the MLX model; upstream's loader drops it, and
   it is inert only because every current config sets it to `false`.
6. Reject or implement `zero_dc` / `freq_range` rather than letting `**kwargs`
   swallow them.
7. MLX chunked inference mirroring `demix_track`'s exact arithmetic, with an
   explicit guard for the `chunk_size % stft_hop_length` assumption §13 depends on.
8. Wire `backend=` through session, separator, `separate_folder`, and CLI.

**Gate**

1. `pip install bs-roformer-infer` remains MLX-free; `import bs_roformer` does
   not import `mlx`.
2. Every stock checkpoint that the MLX path claims to support has a recorded
   Torch-vs-MLX parity fixture within a tolerance derived from measurement, not
   borrowed from upstream (D10).
3. Conversion audit proves 100% parameter match for each supported checkpoint.
4. An unsupported checkpoint under `backend="mlx"` raises a clear error naming
   the reason — never a silent wrong-output run.
5. `cache_info()` returns the same resolver-derived answer under either backend.

</details>

---

## Phase 4 · Variant heads — `doing` (decided: ship 24/24)

Paul's call, 2026-07-30: every variation must be supported. MLX is not missing
these because MLX cannot do them — every primitive was verified present. They are
missing because they are **this project's own additions to the Torch side**, added
in the 2026-07 checkpoint scout, and upstream `mlx-audio-separator` has no reason
to know they exist.

Seam landed (mine): `mlx/heads/__init__.py` owns variant selection through
`build_mask_estimator()`, mirroring Torch's `_create_mask_estimator` signature so
the two cannot drift on what a head receives. `BSRoformerMLX` now takes an explicit
`mask_estimator_variant` instead of swallowing it through `**kwargs` and silently
building the stock head.

**Capability is measured, not declared.** `available_variants()` probes which head
modules exist via `find_spec`, and the backend's `supported_variations()` reads
that — so a head that has not been written cannot be advertised and then fail as an
ImportError deep inside construction.
`tests/test_backends.py::test_every_registry_variation_has_an_mlx_head` asserts the
24/24 goal against the real registry TOML and **is expected to stay red until all
three land**. That red is the honest status, not a broken test.

**Done.** All three heads ported, wired, and verified end to end through the public
session API on their real checkpoints. `test_every_registry_variation_has_an_mlx_head`
is green: MLX covers 24/24.

| Head | Torch LOC | MLX LOC | end-to-end max abs | MLX vs Torch-MPS |
|---|---:|---:|---:|---:|
| `large_inst` | 149 | 195 | 5.364e-07 | **0.5x — slower** |
| `fno` | 255 | 227 | 3.874e-07 | 3.7x |
| `hyperace` | 470 | 601 | 6.258e-07 | 2.8x |

Three findings from the ports, each of which would have shipped silently:

1. **`large_inst`'s checkpoint carries *learned* rotary frequencies.** They differ
   from the theta=10000 default by up to 0.0088 and go negative in places, which no
   fixed formula produces. The vendored `Attention` hard-codes the base and cannot
   express them, and `convert.py` skipped `rotary_embed.freqs` wholesale. Dropping
   them made every key match and computed the wrong thing (0.27 max abs). Also:
   `mx.fast.rope(freqs=...)` wants the *reciprocal* of what
   `rotary_embedding_torch` stores. **Open question this raises: whether any other
   registry checkpoint's trunk rotary frequencies also drifted.** The stock model
   is fine — 3.4e-07 end to end proves it — but the other 19 are unaudited.
2. **The auditing weight loader earned its place.** Wiring `large_inst` into
   `convert.py`, I took an early return that skipped `.gamma` → `.weight`, leaving
   16 norm tensors unmatched. Upstream's `strict=False` would have loaded them as
   random initialisation and produced plausible garbage; the loader named them.
3. **`exact_zero_safe_rfft` is inert inside `fno`** — its `GridEmbedding1D`
   injects nonzero values before the first FFT, so the all-zero Metal artifact
   cannot trigger there. Kept as defence, but honestly not load-bearing the way it
   is in the trunk.

<details><summary>original phase spec</summary>


`large_inst` (149 LOC, reuses existing blocks) → `fno` (255 LOC, needs
`mx.fft.rfft`/`irfft` + complex weights, both verified present) → `hyperace`
(470 LOC, needs `nn.Upsample` semantics matched and NCHW→NHWC weight remap).

Each head ships only with its own parity fixture against the real checkpoint.
Coverage goes 20/24 → 24/24.

</details>

---

## Phase 5 · Documentation and the org record — `todo`

1. `README.md` — backend/device matrix, install story, the arm64-interpreter
   requirement found in evidence §0.
2. `CLAUDE.md` — scope, module layout, verification commands.
3. `CHANGELOG.md` — user-visible changes including D8's contract reversal.
4. A finding into `openmirlab-dev/docs/findings/` carrying this campaign's
   measurements, because the org's 2026-07-21 MLX probe examined `demucs-mlx`
   and does not cover `mlx-audio-separator`.
5. An explicit amendment request against the `ddc-onset-mlx-pilot` Non-goal
   forbidding MLX in another package during the pilot (O2). MLX does not land on
   mainline until that is resolved.
6. Check the public `openmirlab-skills` surface for anything this changes, and
   record the check even if no edit is needed (article 4a).

---

## Open questions that gate phases

| # | Question | Gates |
|---|---|---|
| O1 | ship 24/24 or launch at 20/24 | Phase 4 |
| O2 | how the pilot Non-goal is amended | Phase 3 merge |
| O3 | does `backend="auto"` prefer MLX on Apple Silicon | default only |
| O4 | does a future major flip `device="auto"` to prefer MPS on Mac | not this campaign |

---

## Post-merge follow-ups (recorded 2026-07-31, from the 7-package audit + fable re-plan)

Deliberately NOT done in the campaign window. Each was locked (pin-test or
truthful docstring) instead of fixed, because fixing touches shipped, verified
paths and re-opens article-2 verification per package.

| # | Item | Where | Lock in place today |
|---|---|---|---|
| P1 | Migrate the Torch path onto `ChunkingPlan` | mdxnet (`inference.py:243`), scnet (`runtime.py:60`) | cross-check tests assert inline == plan |
| P2 | Extract the byte-identical SD/SU trunk (666 LOC ×3) verbatim, fixture-gated | scnet `_model/` | `test_trunk_identity.py` pins all three copies |
| P3 | Dedupe `_v2/utils.py` (80-85% reformatted v1 copy; 3-4% genuinely v2) | bandit | dead 25% already deleted; live copy documented |
| P4 | Delete `LinearAttention` + `pack_one`/`unpack_one` (unreachable) | bs-roformer `mlx/attention.py`, `ops.py` | flagged in CLAUDE.md, kept as upstream-shaped surface |
| P5 | Consolidate bs-mamba2's STFT constants into one owner | bs-mamba2 `audio.py` / `mlx_backend.py` | `test_constants_cross_check.py` |
| P6 | Share one streaming-sha256 helper (keep both comparison semantics) | demucs `repo.py` / `checkpoint_runtime.py` | header cross-references explain the split |

USER decisions still pending, untouched: pushing any branch; re-hosting the
dead official-vocals weights URL (NOASSERTION licence review first);
PyPI releases.
