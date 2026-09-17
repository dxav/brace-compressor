# Performance Report

Measurements were taken on 2026-09-17 with Python 3.13, the downloaded
`data/HOAPS_2020-08_6-hourly.nc` sample, and the repository benchmark:

```bash
.venv/bin/python scripts/compress_stats.py \
   --input data/HOAPS_2020-08_6-hourly.nc --variable wvpa --sweep 0.01 0.05 0.2
```

The sample has shape `(124, 320, 720)`, 28,569,600 cells, 67.4% missing
values, and an uncompressed size of 108.98 MB.

| Bound | Compressed | Compression ratio | Reduction | Max error | Encode | Decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 8.90 MB | **12.25x** | 91.8% | 0.01000 | 2.86 s | 1.95 s |
| 0.05 | 6.13 MB | **17.78x** | 94.4% | 0.05000 | 2.15 s | 1.03 s |
| 0.20 | 3.95 MB | **27.60x** | 96.4% | 0.20000 | 1.63 s | 1.04 s |

All three runs reported zero bound violations and an identical missing mask.

## Interpretation

Compression has four major contributors:

1. The spatial-weighted reconstructed-neighbor stencil lowers residual magnitude.
2. The temporal parent adds cross-time correlation when available.
3. Per-block RAW/RANGE/CTX selection chooses the smallest lossless symbol
   representation.
4. Zstandard is attempted independently on the mask and residual payloads and is
   retained only when it reduces size.

The optional Rust extension accelerates both major CPU-heavy stages: the
reconstructed-neighbor scan and the RANGE/CTX rANS loops. Without it, the
Python implementation has the same algorithm, stream format, and semantics
but is slower for large fields. Entropy acceleration is selected independently
of scan acceleration, so a partial or older extension can still use whichever
matching symbols it provides.

These values are benchmark results, not compatibility guarantees. Hardware,
Python version, optional extension availability, and dependency versions can
change timing and a small amount of payload size.
