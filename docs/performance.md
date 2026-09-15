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
