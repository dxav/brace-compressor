# Performance Report (SC-005)

**Date**: 2026-09-15 | **Machine**: single laptop (CPU-only, 4 cores), Python 3.12

## Measured Timings

Benchmark: `tests/conftest.make_smooth_field` synthetic HOAPS-like field
(smooth space-time structure with a land-strip mask), bound = 0.05,
encode + decode round trip, CPU only. Rust causal scan active when available.

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

### Entropy and mask comparison

The following complete-stream measurements use the real HOAPS field above,
outer compression enabled, and the same predictor and quantization settings.
The optimized rANS branch independently evaluates mask and residual payload
compression and selects RLE for this structured missing-value mask. OpenZL and
Zstandard values are from the pluggable-entropy branch using the canonical
bitpacked mask representation.

| Error bound | rANS baseline | rANS + RLE mask | OpenZL | Zstandard |
|---:|---:|---:|---:|---:|
| `0.01` | `12.45×` | **`12.59×`** | `11.57×` | `11.57×` |
| `0.05` | `17.91×` | **`18.21×`** | `17.10×` | `16.87×` |
| `0.2` | `27.38×` | **`28.08×`** | `25.42×` | `24.76×` |

The RLE mask improves rANS by approximately `1.1%`, `1.6%`, and `2.5%`
respectively. All variants preserve the missing mask and respect the error
bound.

### Spatial-weighted causal predictor: +4.2 % CR

The wvpa field has much stronger **spatial** than temporal correlation
(left-neighbor MAE 0.80 vs temporal-parent MAE 2.05). The causal scan's
neighbor weights were rebalanced from `(temporal 4, left 2, top 2,
diag 1, diag 1)` to a **spatial-weighted stencil** `(left 5, top 5,
top-left 2, top-right 2, temporal 1)` in both the Python and Rust scans
(bit-exact). Measured on the real field at bound 0.05:

| Predictor | CR | Max error |
|-----------|----:|----------:|
| Temporal-weighted (old) | 16.34× | 0.0500 ≤ 0.05 |
| **Spatial-weighted (new)** | **17.03×** | 0.0500 ≤ 0.05 |

Residual symbol entropy drops from 5.70 to 5.44 bits/symbol. The Rust and
Python paths remain byte-identical (verified).

### Context-adaptive entropy coder (T048): +5.2 % CR

The entropy stage was upgraded with a **context-adaptive rANS** mode
(`MODE_CTX`). The residual symbols have strong conditional structure:
the first-order conditional entropy is 4.84 bits/symbol vs 5.44
unconditional. The new coder conditions each symbol's probability table
on the previous symbol's magnitude bucket (small/medium/large), which
exploits that structure. Measured on the real field:

| Bound | Old CR (RANGE) | New CR (CTX) | Max error |
|-------|---------------:|-------------:|----------:|
| 0.01  | 11.07× | **12.45×** | 0.0100 ≤ 0.01 |
| 0.02  | 13.70× | **14.34×** | 0.0200 ≤ 0.02 |
| 0.05  | 17.03× | **17.91×** | 0.0500 ≤ 0.05 |
| 0.1   | 20.37× | **21.74×** | 0.1000 ≤ 0.1 |

The encoder builds both candidate payloads (RANGE and CTX) and emits the
smaller; a cheap entropy-based pre-decision skips the RANGE build when
CTX is clearly better. The CTX path is bit-exact (encode/decode
identical) and preserves the hard error bound. Encode/decode timing on
the real field: ~1.6 s / ~2.1 s (vs ~1.9 s / ~2.2 s before).

## Interpretation vs SC-005

- SC-005 requires a standard HOAPS wvpa field (~1–50 MB) to complete "in
  minutes (not hours)". The real 24.6 MB field now round-trips in
  **~4 s total** — far inside the budget.
- Two changes made this possible:
  1. **Rust causal scan** (`rust/hoaps_scan`, PyO3): the dominant
     per-cell loop is ~100× faster than the pure-Python reference
     (bit-exact, so streams are byte-identical either way).
    2. **Deterministic cold-start prior**: the scan avoids model inference and
      keeps runtime dominated by the causal traversal and entropy coding.

## Known Optimization Headroom (not required for v1 acceptance)

- The Rust scan is single-threaded; parallelizing over time slices
  (each slice's scan is independent given the previous slice's state)
  would give a further multi-core speedup.

## Guarantees unchanged by performance work

- SC-001: max abs error verified ≤ bound on every round trip.
- SC-002: missing mask bit-identical (FR-004/FR-015).
- SC-003: ≥ 2× (measured 11.67× on real data, 4.51× on synthetic).
- SC-004: CR monotonic non-increasing for looser bounds (test suite).
- SC-007: space-time ≥ spatial-only baseline (test suite, 5% tolerance).
