# Development Guide

## Requirements

- Python 3.11 or newer
- A working C/Rust toolchain only when building the optional Rust extension
- `numpy`, `numcodecs`, and the test extras from `pyproject.toml`

The Python implementation is always the correctness fallback. A Rust build is
an optimization, not a prerequisite for using the codec.

## Install from a checkout

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The package uses `maturin` as its build backend because the optional
`hoaps_scan` extension is part of the distribution. If a local Rust toolchain
is unavailable, install the Python dependencies directly and run from the
checkout with `PYTHONPATH=src`.

## Build the Rust accelerator

```bash
cargo build --manifest-path rust/hoaps_scan/Cargo.toml --release
python -m pip install -e .
```

The extension exports `causal_scan_encode` and `causal_scan_decode`. Tests
compare behavior through the public codec; the Python fallback remains the
reference implementation.

## Tests

From the repository root:

```bash
PYTHONPATH=. pytest tests -q
PYTHONPATH=. pytest tests/unit -q
PYTHONPATH=. pytest tests/integration -q
```

`PYTHONPATH=.` avoids a name collision with an unrelated installed package
called `tests` on some environments. The expected current result is 68 tests
when running the full suite.

The test groups are:

- `tests/unit`: bounds, quantization, masks, container framing, and entropy;
- `tests/contract`: `numcodecs` API, registry, config, and output buffers;
- `tests/integration`: error guarantees, missing values, bound sweeps, and
  space-time compression.

## Benchmarks

Synthetic field:

```bash
.venv/bin/python scripts/compress_stats.py \
  --shape 8 90 180 --bound 0.05
```

Real NumPy field:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/wvpa_2020-08-01_07.npy --bound 0.05
```

Bound sweep:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/wvpa_2020-08-01_07.npy --sweep 0.01 0.05 0.2
```

The benchmark independently checks maximum valid-value error, mask identity,
RMSE, timing, container metrics, and compression ratio. `--json` emits data
for automation. `scripts/analyze_entropy.py` is an exploratory diagnostic for
residual-symbol distributions; it is not required by the codec runtime.

## Public API

```python
from hoaps_compressor import HoapsWvpaCodec

codec = HoapsWvpaCodec(
    shape=(time, latitude, longitude),
    error_bound=0.05,
    missing_value="nan",
)
encoded: bytes = codec.encode(field)
decoded = codec.decode(encoded)
config = codec.get_config()
restored = HoapsWvpaCodec.from_config(config)
```

`field` must represent float32 values with the configured shape. `decode` can
write into a C-contiguous writable float32 `out` array of the same byte size.
The codec registers itself under `hoaps-wvpa` when `hoaps_compressor` is
imported.

## Change discipline

Keep encode and decode changes paired. Any predictor arithmetic, scan order,
quantization rule, entropy layout, or container flag change can affect stream
compatibility and must update `MODEL_VERSION`, tests, and the algorithm/format
documentation as appropriate. Do not use original field values in decode-time
prediction. Run the full suite and a real-data benchmark before changing the
reported performance numbers.
