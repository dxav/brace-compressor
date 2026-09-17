# Development Guide

## Requirements

- Python 3.11 or newer
- A C/Rust toolchain only when building the optional Rust extension
- `numpy`, `numcodecs`, and the test extras from `pyproject.toml`

The Python implementation is always the correctness fallback. Rust is an
optimization, not a prerequisite for using the codec.

## Install From A Checkout

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The package uses `maturin` for the optional `brace_scan` extension. Without a
Rust toolchain, install the Python dependencies and run from the checkout with
`PYTHONPATH=src`.

## Build The Rust Accelerator

```bash
cargo build --manifest-path rust/brace_scan/Cargo.toml --release
python -m pip install -e .
```

The extension contains the float32/float64 causal scan and static/context
adaptive rANS implementations. The codec selects matching Rust functions
automatically and falls back to Python when unavailable. Both implementations
use the same stream format and arithmetic.

## Tests

From the repository root:

```bash
PYTHONPATH=src pytest tests -q
PYTHONPATH=src pytest tests/unit -q
PYTHONPATH=src pytest tests/integration -q
```

The current full suite contains 113 tests. Recommendation tests cover all
supported requirement families, nested `any`/`all` plans, CR-aware candidate
selection, aggregate repairs, local quadratic steps, lossless byte transforms,
and malformed metadata cases.

Test groups:

- `tests/unit`: bounds, quantization, masks, containers, entropy, constraints,
  and recommendation planning;
- `tests/contract`: `numcodecs` API, registry, config, and output buffers;
- `tests/integration`: error guarantees, missing values, bound sweeps, and
  space-time compression.

## Benchmarks

Synthetic field:

```bash
PYTHONPATH=src python scripts/compress_stats.py \
  --shape 8 90 180 --bound 0.05
```

The benchmark enables lossless outer Zstandard compression by default. Use
`--no-outer-compress` to measure the entropy-coded container without that final
pass. The equivalent setting is `BraceCodec(..., outer_compress=False)`.

HOAPS NetCDF field:

```bash
python -m pip install -e ".[analysis]"
PYTHONPATH=src python scripts/compress_stats.py \
  --input data/HOAPS_2020-08_6-hourly.nc --variable wvpa --bound 0.05
```

Bound sweep:

```bash
PYTHONPATH=src python scripts/compress_stats.py \
  --input data/HOAPS_2020-08_6-hourly.nc --variable wvpa \
  --sweep 0.01 0.05 0.2
```

Typed recommendation benchmark:

```bash
PYTHONPATH=src python scripts/compress_era5_with_recommendations.py \
  --input data/era5_pressure_20260715T1200_4levels.nc \
  --output data/era5_pressure_recommendation_example.json
```

This benchmark encodes and decodes every selected variable, records the full
selected requirement tree and diagnostics, reports the winning strategy and
repair count, and exits nonzero if an active requirement fails. For an `any`
plan, complete candidate streams are compared by final encoded size. Scalar
reference metrics are reported separately when they do not describe the
selected branch.

The regular benchmark independently checks maximum valid-value error, mask
identity, RMSE, timing, container metrics, and compression ratio. With
`--json`, it emits stage timings for mask handling, scan, verification, entropy,
container work, and decode restoration. The same timing data is available in
`codec.last_timings`.

## Public API

```python
from brace_compressor import BraceCodec

codec = BraceCodec(
    shape=(time, latitude, longitude),
    error_bound=0.05,
    missing_value="nan",
)
encoded: bytes = codec.encode(field)
decoded = codec.decode(encoded)
config = codec.get_config()
restored = BraceCodec.from_config(config)
```

`BraceCodec` inherits from `numcodecs.abc.Codec` and registers as `brace` when
`brace_compressor` is imported. The configured dtype must be `float32` or
`float64`; `decode` can write into a compatible writable output array.

## Change Discipline

Keep encode and decode changes paired. Predictor arithmetic, scan order,
quantization rules, entropy layout, container flags, or recommendation schema
changes must update model/schema validation, tests, and the algorithm and
format documentation. Do not use original field values in decode-time
prediction. Run the full suite, static diagnostics, `git diff --check`, and a
real-data benchmark before changing reported performance numbers.
