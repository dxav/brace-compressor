# Performance Report

Measurements were taken on 2026-09-16 with Python 3.12, the bundled
`data/wvpa_2020-08-01_07.npy` sample, and the repository benchmark:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/wvpa_2020-08-01_07.npy --sweep 0.01 0.05 0.2
```

The sample has shape `(28, 320, 720)`, 6,451,200 cells, 68.7% missing
values, and an uncompressed size of 24.61 MiB.

| Bound | Compressed | Compression ratio | Reduction | Max error | Encode | Decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.93 MiB | **12.75x** | 92.2% | 0.01000 | 1.77 s | 2.61 s |
| 0.05 | 1.33 MiB | **18.53x** | 94.6% | 0.05000 | 1.45 s | 1.47 s |
| 0.20 | 877 KiB | **28.74x** | 96.5% | 0.20000 | 0.79 s | 1.10 s |

All three runs reported zero bound violations and an identical missing mask.
The looser bounds produce non-decreasing compression ratios because the
quantization step is `2 * bound`.

## Interpretation

Compression has four major contributors:

1. The spatial-weighted reconstructed-neighbor stencil lowers residual magnitude.
2. The temporal parent adds cross-time correlation when available.
3. Per-block RAW/RANGE/CTX selection chooses the smallest lossless symbol
   representation.
4. Zstandard is attempted independently on the mask and residual payloads and is
   retained only when it reduces size.

The optional Rust extension accelerates the linear scan. Without it,
the Python implementation has the same algorithm and stream semantics but is
slower for large fields.

These values are benchmark results, not compatibility guarantees. Hardware,
Python version, optional extension availability, and dependency versions can
change timing and a small amount of payload size.
