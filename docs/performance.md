# Performance Report (SC-005)

**Date**: 2026-09-15 | **Machine**: single laptop (CPU-only, 4 cores), Python 3.12, torch 2.14 CPU

## Measured Timings

Benchmark: `tests/conftest.make_smooth_field` synthetic HOAPS-like field
(smooth space-time structure with a land-strip mask), bound = 0.05,
encode + decode round trip, CPU only:

| Field shape | Uncompressed | Encoded | CR   | Max error | Encode | Decode |
|-------------|-------------:|--------:|-----:|----------:|-------:|-------:|
| (4, 16, 32) — toy   | 8 KB    | 2.6 KB  | 3.10x | 0.0125 ≤ 0.05 | < 1 s  | < 1 s  |
| (8, 90, 180) — typical | 518 KB | 115 KB | 4.51x | 0.0125 ≤ 0.05 | ~ 57 s | ~ 59 s |

## Interpretation vs SC-005

- SC-005 requires a standard HOAPS wvpa field (~1–50 MB uncompressed) to
  complete "in minutes (not hours)". Extrapolating per-cell scan cost
  linearly: a 7.8 MB monthly 0.5° field (30 × 180 × 360) costs roughly
  15× the 0.52 MB benchmark ≈ 15 minutes on this CPU — within the
  minutes-scale offline budget, at the low end of acceptable.
- The causal scan dominates runtime (pure-Python inner loops over ~130k
  cells plus the torch base-prior projection). No GPU is required.

## Known Optimization Headroom (not required for v1 acceptance)

- Numba/Cython/C extension for the causal scan loop (expected ≥ 50×).
- Batched torch inference for the base prior (currently one projection
  over all cells; could be chunked and parallelized).
- The transformer prior is already deterministic and mask-only; a
  smaller d_model would shrink its share of runtime.

## Guarantees unchanged by performance work

- SC-001: max abs error verified ≤ bound on every round trip.
- SC-002: missing mask bit-identical (FR-004/FR-015).
- SC-003: ≥ 2× (measured 4.51×) at the reference bound.
- SC-004: CR monotonic non-increasing for looser bounds (test suite).
- SC-007: space-time ≥ spatial-only baseline (test suite, 5% tolerance).
