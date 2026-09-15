# Performance Report (SC-005)

**Date**: 2026-09-15 | **Machine**: single laptop (CPU-only, 4 cores), Python 3.12, torch 2.14 CPU

## Measured Timings

Benchmark: `tests/conftest.make_smooth_field` synthetic HOAPS-like field
(smooth space-time structure with a land-strip mask), bound = 0.05,
encode + decode round trip, CPU only. **Rust causal scan + downsampled
transformer prior active.**

| Field shape | Uncompressed | Encoded | CR   | Max error | Encode | Decode |
|-------------|-------------:|--------:|-----:|----------:|-------:|-------:|
| (4, 16, 32) — toy   | 8 KB    | 2.6 KB  | 3.10x | 0.0125 ≤ 0.05 | < 0.1 s | < 0.1 s |
| (8, 90, 180) — typical | 518 KB | 115 KB | 4.51x | 0.0125 ≤ 0.05 | ~ 0.1 s | ~ 0.1 s |

### Real HOAPS data (2020-08-01..07, 6-hourly, 28×320×720)

Downloaded from the ESIWACE object store, `wvpa` variable, fill value
`-9e+33` → NaN, bound = 0.05:

| Metric | Value |
|--------|-------|
| Uncompressed | 24.61 MB (6.45 M cells, 68.7 % missing) |
| Compressed | 2.11 MB |
| **Compression ratio** | **11.67×** (91.4 % smaller, 2.74 bits/value) |
| Max abs error | 0.0125 ≤ 0.05 (0 violations) |
| RMSE / bias | 0.0072 / −0.000001 |
| Missing mask | identical (SC-002) |
| Encode / decode | **1.93 s / 2.19 s** (13.4 MB/s) |

### Enhancement phase (T039/T044): trained base prior + block-local causal predictor

**Corrected assessment.** The earlier claim that training the prior raised
CR from 11.67× to 16.34× was **wrong**: it compared the old quantization
(`Δ = bound/2`, max error 0.0125) against the new quantization
(`Δ = 2·bound`, max error 0.05). The CR jump was entirely due to the
quantization-step change, **not** the trained prior.

Measured at the **same** quantization step (`Δ = 2·bound`), the random and
trained priors give **identical CR**:

| Bound | Random prior CR | Trained prior CR | Max error |
|-------|----------------:|-----------------:|----------:|
| 0.05  | 16.34× | 16.34× | 0.0500 ≤ 0.05 |
| 0.01  | 11.07× | 11.07× | 0.0100 ≤ 0.01 |

The trained prior provides **no CR benefit** at any bound. This is
expected: the base prior only affects **cold-start cells** (the first
valid cell of each scan region, ~10k of 2.02 M valid cells, 0.5 %), which
contribute negligibly to residual entropy. The causal scan's weighted
average dominates prediction.

Moreover, the trained prior is **slightly worse** at cold-start prediction
than the random prior:

| Prior | Cold-start MAE | Cold-start RMSE |
|-------|---------------:|----------------:|
| Random | 15.70 | 18.56 |
| Trained | 16.23 | 20.30 |

The masked-cell training (loss 212→207 on the coarse grid) did not
improve the prior's ability to predict actual wvpa values. **Conclusion:
the trained prior adds no value and is not worth shipping as the default.**
The quantization-step change (`Δ = 2·bound`) is the real CR driver and is
already in place.

### Block-local causal predictor (T040–T043): experimental, opt-in

A block-local causal attention predictor (`use_block_predictor=True`) is
implemented and integrated into the causal scan. It is **correct and
deterministic** (encode/decode bit-identical, hard bound preserved), but
**currently experimental**: its `block_in_proj` weights are untrained, so
on smooth fields it predicts worse than the fixed weighted average and
lowers CR (e.g. 3.11× vs 4.18× on the synthetic field). It also incurs a
per-block transformer forward pass, which is slow on the full 6.45 M-cell
field. It is therefore **opt-in (default off)** and requires training of
the block predictor head to be beneficial. The Rust port (T043) is
deferred until the trained block predictor demonstrates a CR win.

## Interpretation vs SC-005

- SC-005 requires a standard HOAPS wvpa field (~1–50 MB) to complete "in
  minutes (not hours)". The real 24.6 MB field now round-trips in
  **~4 s total** — far inside the budget.
- Two changes made this possible:
  1. **Rust causal scan** (`rust/hoaps_scan`, PyO3): the dominant
     per-cell loop is ~100× faster than the pure-Python reference
     (bit-exact, so streams are byte-identical either way).
  2. **Downsampled transformer prior** (`base_prior`): self-attention is
     O(N²), so the prior now runs on a coarse grid (≤ 8192 tokens) and
     upsamples, cutting the prior from effectively-infinite to ~0.5 s on
     the real field.

## Known Optimization Headroom (not required for v1 acceptance)

- The Rust scan is single-threaded; parallelizing over time slices
  (each slice's scan is independent given the previous slice's state)
  would give a further multi-core speedup.
- The transformer prior could use linear/local attention instead of
  downsampling for even finer priors at large scale.
- A smaller `d_model` would shrink the prior's share of runtime.

## Guarantees unchanged by performance work

- SC-001: max abs error verified ≤ bound on every round trip.
- SC-002: missing mask bit-identical (FR-004/FR-015).
- SC-003: ≥ 2× (measured 11.67× on real data, 4.51× on synthetic).
- SC-004: CR monotonic non-increasing for looser bounds (test suite).
- SC-007: space-time ≥ spatial-only baseline (test suite, 5% tolerance).
