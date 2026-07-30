# Evidence — measured, 2026-07-30

Every number below was produced on one machine with disposable probes outside
the package. No product source was modified to obtain them.

**Rig**: Apple M2, 24 GB unified memory, macOS 26.5.1, Python 3.12.12 (arm64),
torch 2.13.0, mlx 0.31.2, mlx-audio-separator 0.1.5.

## 0. Environment trap found first

`uv` was installed from Intel Homebrew (`/usr/local`), so the binary itself was
x86_64 and resolved x86_64 interpreters. A plain `uv sync` therefore produced an
environment where `torch.backends.mps.is_available()` is structurally `False` —
MPS is invisible, not broken. Replaced with the native arm64 build
(`uv 0.12.0 aarch64-apple-darwin`).

**Implication for the package**: a Mac user following `CLAUDE.md`'s documented
`uv sync` can land in an environment where the MPS path silently cannot exist.
Whatever we ship must state the arm64 interpreter requirement.

## 1. MPS operator coverage — 12/12, no fallback

Run with `PYTORCH_ENABLE_MPS_FALLBACK=0`, so any operator lacking an MPS kernel
raises rather than silently relocating to CPU. Shapes mirror a real chunk
(stereo, `n_fft=2048`, `hop=441`).

| Operator | Source site | MPS | CPU-vs-MPS max abs err |
|---|---|---|---|
| `torch.hann_window(device=)` | `bs_roformer.py:521` | OK | 5.960e-08 |
| `torch.stft(return_complex=True)` | `bs_roformer.py:523` | OK | 3.433e-05 |
| `torch.view_as_real` | `bs_roformer.py:524` | OK | 3.433e-05 |
| `torch.view_as_complex` | `bs_roformer.py:575` | OK | 0.000e+00 |
| complex multiply (mask application) | `bs_roformer.py:578` | OK | 9.155e-05 |
| `F.pad` on a complex tensor | `bs_roformer.py:584` | OK | 9.155e-05 |
| `index_fill` on complex (`zero_dc`) | `bs_roformer.py:588` | OK | 9.155e-05 |
| `torch.istft` | `bs_roformer.py:590` | OK | 1.118e-06 |
| `scaled_dot_product_attention` | `attend.py:94` | OK | 9.537e-07 |
| SDPA under `torch.backends.cuda.sdp_kernel` | `attend.py:93` | OK | 9.537e-07 |
| `fft.rfftn` → complex mul → `irfftn` | `fno.py:123-140` | OK | 1.371e-06 |

Complex arithmetic and FFT — the operators predicted most likely to be missing —
are all present in torch 2.13.

## 2. Full Torch model, CPU vs MPS

Real architecture from the bundled `BS-Rofo-SW-Fixed.yaml` (dim 256, depth 12,
6 stems, `flash_attn=True`), 174.7 M parameters, random weights, 2 s stereo.

```
[cpu] forward OK   finite=True
[mps] forward OK   finite=True
CPU vs MPS   max_abs=6.612e-08  mean_abs=1.007e-08  rel_to_peak=1.285e-06
```

`demix_track()` — the real chunked overlap-add path — completed on MPS and
returned all six stems.

## 3. Throughput at the config's real chunk size

`chunk_size=588800` (13.35 s), `num_overlap=2`, so each forward advances 6.68 s
of output.

| Path | per chunk | realtime factor | 1 min of audio |
|---|---:|---:|---:|
| Torch CPU | 90.66 s | 0.07x | ~13.6 min |
| Torch MPS | 26.61 s | 0.25x | ~4.0 min |

**3.4x**, MPS peak allocation 5.34 GB.

## 4. MLX vs Torch-MPS, real checkpoint

The package's own sha256-verified `BS-Rofo-SW-Fixed.ckpt` (699 412 152 bytes)
and its config were fed directly to `mlx-audio-separator`'s `BSRoformerMLX`.
10 s stereo, same input both sides.

**Weight conversion is complete**:

```
torch state_dict tensors : 1939
converted to MLX         : 1915   (24 skipped = rotary_embed.freqs buffers, deliberate)
MLX model expects        : 1915
MATCHED (actually loaded): 1915
  dropped by strict=False: 0
  left at random init    : 0
```

This matters because `load_roformer_model` calls `load_weights(..., strict=False)`,
which silently discards unmatched keys. For this model nothing was discarded — but
the check must be re-run per checkpoint, not assumed.

**Speed and parity**, with `MLX_USE_FAST_SDP=1 MLX_ENABLE_COMPILE=1`:

| AMP | MLX forward | vs Torch-MPS | rel_L2 vs Torch | max_abs |
|---|---:|---:|---:|---:|
| off | 10.07 s | **2.17x** | **7.27e-07** | 3.87e-07 |
| on | 9.98 s | 1.93x | 6.35e-05 | 3.79e-05 |

AMP bought no measurable speed on this path and degraded parity by ~87x.
Torch-MPS baseline varied 19.25–22.03 s across runs (±7%), so the honest speedup
band is **1.9–2.2x**.

For scale: `mlx-audio-separator`'s own release gate accepts `rel_L2 <= 5e-2`.
What we measured is five orders of magnitude tighter.

## 5. Output length divergence — diagnosed, shallow

Torch returns 440 832 samples for a 441 000-sample input; MLX returns 441 000.

```
input 441000  frames 862  istft out 440832
(frames-1)*hop = 440832        shortfall = 168
```

`torch.istft` returns exactly `(frames-1) * hop` and drops the tail; MLX restores
the original length. A trim/pad convention, not a numerical disagreement.
`demix_track` already slices `x[..., :length]`, so the package layer may absorb
this — **to be confirmed, not assumed**.

## 6. Variant-head portability — primitives all present

The three variant mask-estimator heads and whether MLX has what they need:

| Head | LOC | Distinctive ops | MLX availability |
|---|---:|---|---|
| `large_inst.py` | 149 | `GLU`, `cat`, `Sequential`, `Identity` | all present; reuses transformer blocks MLX already has |
| `fno.py` | 255 | `fft.rfftn`/`irfftn`, complex weights, `GELU`, `Conv1d` | all present (`mx.fft.rfft`/`irfft`, `complex64` verified) |
| `hyperace.py` | 470 | `F.interpolate` ×12, `InstanceNorm2d`, `Conv2d`, `bmm` | all present (`nn.Upsample`, `nn.InstanceNorm`, `nn.Conv2d`, `mx.matmul`) |

Checked availability: 11/11 `mlx.nn` layers, 5/5 `mlx.core` ops, `mx.fft.rfft` /
`mx.fft.irfft` / `complex64` complex multiply — no gaps.

Two known porting costs, both documented rather than novel: MLX `Conv2d` is NHWC
against Torch's NCHW (weights need `OIHW → OHWI`, the same remap the org's DDC
probe recorded), and `F.interpolate` → `nn.Upsample` needs mode/`align_corners`
semantics matched explicitly.

## 7. Dependency surface of the MLX route

| Package | Needed by the roformer model? | License | Version spec | Compiled |
|---|---|---|---|---|
| `mlx` | yes | MIT | — | yes (`mlx-metal` wheel) |
| `mlx-spectro` | **yes** (`bs_roformer.py:27,858`) | MIT | `mlx>=0.30.3` (floor) | no — 6 072 lines pure Python |
| `mlx-audio-io` | **no** — zero imports on the roformer path | MIT | `mlx==0.31.2` (**exact pin**) | **yes** (AudioToolbox backend) |

Installing `mlx-audio-separator` pulls `mlx-audio-io` transitively; vendoring only
the roformer model files does not.

## 8. Registry coverage

24 registry models: 20 stock, 2 `hyperace`, 1 `fno`, 1 `large_inst`. Without the
three ported heads an MLX backend covers 20/24.

## 9. All 24 registry configs instantiate under MLX — and 4 of them silently lie

Delegated sweep (all 24 config YAMLs fetched from their registry URLs, each
sha256-verified against `checkpoints.toml`, then parsed with the package's own
`SafeLoaderWithTuple` and used to construct a real `BSRoformerMLX` with
parameters materialized via `mx.eval`).

**24/24 construct without raising.** That result is worse news than it sounds:

- The 20 stock models build the correct architecture.
- The 4 variant models (`hyperace` ×2, `fno`, `large_inst`) **also build without
  error, and build the wrong model**. `BSRoformerMLX.__init__` swallows unknown
  arguments through `**kwargs` and always constructs the plain `MaskEstimator`;
  there is no variant branch anywhere in the MLX code. `large_inst` is worse
  still — its four extra time/frequency Transformer pairs have no constructor
  parameter to reach at all.

This is architecture failure mode 3 in its purest form: no exception, no warning,
a structurally different network. Combined with upstream's
`load_weights(strict=False)`, a variant checkpoint would load partially and
produce plausible-looking garbage. **The MLX backend must reject a variant
checkpoint explicitly until its head exists.**

Two latent defects also surfaced, currently inert for this registry:

- `create_bs_roformer_mlx` (`loader.py:71-73`) forwards `stft_n_fft`,
  `stft_hop_length`, `stft_win_length` but **not** `stft_normalized`, even though
  the model class accepts it. Harmless only because all 24 configs set it to
  `false`, which matches the class default.
- `zero_dc` and `freq_range` exist on the Torch side and have no MLX counterpart;
  no current config sets either, so `**kwargs` swallows nothing today.

## 10. The packaged config is not the config the registry describes

Found while reconciling a contradiction between two probes. Unrelated to MLX;
recording it because it is a live provenance gap.

For the default model, `checkpoints.toml` records the config artifact as
sha256 `f9fada9f…`, 4613 bytes. The file the runtime actually loads is the
packaged `src/bs_roformer/configs/BS-Rofo-SW-Fixed.yaml` — 686 bytes, sha256
`52df622c…` — because config resolution prefers the packaged copy over the URL.
The cached copy under `~/.cache/` is byte-identical to the packaged one.

The difference is real: the packaged config omits `freqs_per_bands` entirely,
while the registry's config states it.

**Verified equivalent, not merely assumed**: the URL config's `freqs_per_bands`
has 62 entries summing to 1025, exactly matching `BSRoformer`'s own default. The
two therefore build identical tensor shapes, which is also why the real
checkpoint strict-loads with 0 missing and 0 unexpected keys.

So: not an accuracy bug, but the registry's recorded integrity metadata describes
a file the package does not use. Out of scope for this campaign; worth its own
follow-up under article 4's provenance rule.

## 11. Real-checkpoint CPU-vs-MPS parity, through the package's own path

Default checkpoint, `demix_track()` unmodified, 5 s seeded stereo (seed
20260730), torch 2.13.0 on M2:

| stem | max_abs | mean_abs | rel_to_peak |
|---|---:|---:|---:|
| bass | 2.401e-10 | 2.107e-11 | 5.097e-06 |
| drums | 1.219e-10 | 1.430e-11 | 3.532e-06 |
| other | **1.136e-07** | 4.068e-09 | 4.463e-06 |
| vocals | 1.572e-09 | 6.641e-11 | 6.289e-06 |
| guitar | 7.094e-11 | 7.387e-12 | 3.361e-06 |
| piano | 1.048e-09 | 2.840e-11 | 5.750e-06 |

This is now `tests/test_device_parity.py`, marked `realweights`, asserting
`max_abs < 1e-5` — roughly two orders of headroom over what was measured, which
is wide enough not to flake and far too tight for a genuinely broken MPS path to
slip through.

**Timing from that particular run is discarded.** It reported CPU 224.7 s against
MPS 28.2 s (8.0x), but a delegated benchmark was running concurrently on the same
machine and inflated the CPU leg. The MPS number matches the clean run (26.6 s),
the CPU number does not (90.7 s). The parity numbers are unaffected — they compare
outputs, not clocks — but §3's 3.4x remains the number of record.

## 12. MLX at the production chunk size, and through the real chunked path

Delegated measurement, same machine, `MLX_USE_FAST_SDP=1 MLX_ENABLE_COMPILE=1
MLX_ENABLE_AMP=0`, real checkpoint.

**Single warm forward at `chunk_size=588800`:**

| Path | per chunk | peak memory | 1 min of audio |
|---|---:|---:|---:|
| Torch MPS | 26.61 s | 5.34 GB | ~4.0 min |
| MLX | 15.02 s | **2.70 GB** | ~2.25 min |

**1.77x** — narrower than the 1.9–2.2x measured on a 10 s buffer. The wider figure
does not survive the production chunk size, so 1.77x is the honest headline. The
halved memory is a second, separate win worth stating: 2.70 GB against 5.34 GB
changes which Macs can run this at all.

**Through a faithful mirror of `demix_track`'s chunking** (20 s audio, 5 chunks,
same chunk/step/fade/border/window/counter arithmetic): Torch 24.92 s/chunk, MLX
12.60 s/chunk, ≈1.98x.

Unexplained and recorded as such: MLX's in-loop per-chunk average (12.60 s) is
*faster* than its isolated warm forward at the same size (15.02 s). Plausibly
Metal buffer reuse or compile-cache effects across repeated same-shape calls, but
that is a hypothesis, not a measurement.

**Cross-backend parity through the full chunked path** (Torch-MPS vs MLX):

| stem | max_abs | rel_L2 |
|---|---:|---:|
| bass | 2.59e-10 | 3.96e-06 |
| drums | 3.91e-06 | 6.02e-06 |
| other | 4.77e-07 | 1.95e-06 |
| vocals | 1.68e-10 | 6.35e-06 |
| guitar | 2.09e-07 | 1.10e-06 |
| piano | 8.82e-11 | 4.04e-06 |

Two independent implementations agreeing to ~1e-06 relative L2 is stronger than a
reimplementation has any right to be, and is worth treating as a claim to re-test
on real music rather than a settled fact.

## 13. The 168-sample divergence does not reach the stems — for one specific reason

Both backends returned exactly 882 000 samples for an 882 000-sample input, all
six stems. The reason is arithmetic, not robustness: `chunk_size=588800` is
exactly `1150 × stft_hop_length(512)`, so Torch's `istft` reconstructs precisely
`C` samples per chunk and `demix_track`'s `x[..., :length]` slice never sees a
mismatch. MLX passes an explicit output length and matches by construction.

**Conditional, and the condition is undocumented.** A future config whose
`chunk_size` is not divisible by `stft_hop_length` would make Torch's per-chunk
`istft` return a different length mid-loop, which the code as written would hit
as a hard crash rather than a silent trim. Inferred from reading `demix_track`,
not measured. Worth an explicit guard.

## 14. BLOCKING — silence makes Torch and MLX diverge, and it contaminates the signal

Found while integration-testing the MLX backend end to end. This invalidates the
comfortable reading of §12's parity table.

Single forward, real checkpoint, `chunk_size=588800`, stereo:

| Input | max abs error, Torch vs MLX |
|---|---:|
| 588800 samples of Gaussian noise, no silence | **4.023e-07** |
| 220500 samples of noise + 368300 zeros | **1.455e-02** |
| noise + tail scaled by 1e-6 (near-silent) | **3.358e-04** |

Two things make this serious rather than cosmetic:

1. **The divergence scales with how close the tail is to silence** — so it is a
   numerical instability around zero, not a one-off.
2. **The error lands in the signal region, not the silent one.** For the
   zero-padded case, the worst error over the first 220 500 samples is 1.455e-02;
   over the padded tail it is 9.098e-04. Silence at the *end* corrupts the output
   at the *beginning*, which means something couples across the whole time axis.

**Why §12 missed it.** That measurement used 20 s of continuous Gaussian noise.
No silence, no trigger, and the result read as "surprisingly strong agreement".
It was measuring the easy case. Real music is worse than noise here: tracks have
intros, outros, and rests, and — unavoidably — **every track's final chunk is
zero- or reflect-padded**.

Ruled out already: the chunked overlap-add (full `separate()` error exactly equals
the single-forward error), weight conversion (1915/1915 audited), and the
normalization layer (`mlx/model.py:261-282` is mathematically identical to
`bs_roformer.py:58-65`; the `mx.fast.rms_norm` fast path is env-gated off).

### Root cause — found, fixed, and guarded

A three-link chain, each link measured rather than argued:

1. **MLX 0.31.2's Metal `rfft` kernel.** It packs two real FFTs into one complex
   FFT and recovers each by conjugate symmetry. In float32 on GPU that
   cancellation is not bit-exact, so a frame whose true value is exactly zero
   returns roughly `4.5e-07`. Verified GPU-specific: the same call on MLX's CPU
   stream returns exact `0.0`, every time. Verified pairing-specific: the leak
   appears only in a zero frame batched alongside a nonzero partner.
2. **`L2Norm` amplifies it about a millionfold.** The layer discards magnitude by
   construction, and its `eps` of `1e-12` sits five orders below the artifact, so
   the clamp never engages. A band vector whose true norm is 0.403 came back as
   1.872 — pure noise normalized into a full-scale, essentially random direction.
3. **Time-axis attention spreads it everywhere.** Attention runs over all 1151
   time positions per band, so one corrupted frame reaches every query position.
   That is the answer to why silence at the *end* corrupted the *start*.

**Fix**: route `mx.fft.rfft` through the CPU stream for the duration of the STFT
(`mlx/model.py::exact_zero_safe_rfft`). Measured after: `2.226e-07` on the
zero-padded chunk, `2.831e-07` with no silence, `1.937e-07` near-silent — one
noise floor for all three, and **no speed cost** (10.5 s/chunk against 11.9 s
before).

**Raising `eps` was considered and rejected.** Genuinely quiet audio has
legitimate band norms in the same `1e-6` range, so a larger clamp would trade a
loud bug for a quiet one that only shows on soft material — worse, because it
would not announce itself.

**Guarded**: `tests/test_mlx_parity.py` covers signal, zero-padded, and
near-silent tails. Validated by removing the workaround and confirming the two
silent cases fail (`1.455e-02` and `6.616e-05`) while the clean case passes — a
regression test that passes without the fix protects nothing, and the first
attempt at this validation silently failed to apply the edit, which is exactly how
one ships a decorative test.

**Status: resolved.** End to end through the public session API on the real
checkpoint, all seven outputs agree with Torch to `3.427e-07`, in 9.3 s against
Torch-MPS's 25 s.

## Not yet measured

These are open and must not be stated as facts until probed:

1. Whether the 20 stock checkpoints all **convert and strict-match**. Only the
   default SW model was verified at checkpoint level; §9 verified all 24 at
   *config* level, which is a weaker claim.
2. Whether the three variant heads reproduce Torch output once ported.
3. **Real-audio parity.** Everything above used seeded Gaussian noise. Music has
   structure that noise does not, and a parity result on noise is evidence, not
   proof.
4. Whether MLX's memory advantage (§12) holds for the memory-heavy checkpoints —
   the MVSep Mega 53-stem model in particular, which upstream says wants 16 GB.
5. Whether any registry config has a `chunk_size` that is not a multiple of its
   `stft_hop_length` (§13's unguarded condition).
