# BRACE

[![CI](https://github.com/eobytes/brace-compressor/actions/workflows/ci.yml/badge.svg)](https://github.com/eobytes/brace-compressor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/optional%20accelerator-Rust-orange.svg?logo=rust)](rust/brace_scan)

**BRACE** (*Bounded Residual Adaptive Compression Engine*) is an error-bounded,
lossless-symbol codec for climate data. It compresses two-dimensional
`(latitude, longitude)` slices or three-dimensional float32 arrays with shape
`(time, latitude, longitude)` using a deterministic reconstructed-neighbor
predictor, bound-derived residual quantization, and lossless entropy coding.

## Guarantees

- For `error_bound > 0`, every valid decoded value is verified against the
  configured absolute bound before the stream is returned.
- Missing positions are stored in an independent lossless mask and restored
  without loss.
- Entropy coding is bit-exact, so decoded symbols match encoded symbols.
- Rust and Python scans use the same order and arithmetic.
- `error_bound=0` uses the same quantized path as positive bounds, with
  `float32` machine epsilon as the minimum effective error budget. It is
  therefore near-lossless but can differ by floating-point computation error.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

The package is built with `maturin` and can include the optional Rust
accelerator. The codec still works without a Rust toolchain by using the
Python scan.

## Use the codec

```python
import numpy as np
from brace_compressor import BraceCodec

shape = (8, 90, 180)
field = np.zeros(shape, dtype="float32")
field[:, :8, :] = np.nan

codec = BraceCodec(shape=shape, error_bound=0.05)
encoded = codec.encode(field)
decoded = codec.decode(encoded)

assert decoded.shape == shape
assert np.array_equal(np.isnan(decoded), np.isnan(field))

config = codec.get_config()
codec_copy = BraceCodec.from_config(config)
```

The class implements `codec_id`, `encode`, `decode`, `get_config`, and
`from_config` for `numcodecs` integration. Importing `brace_compressor`
registers the `brace-wvpa` codec with the registry.

## Run tests

```bash
PYTHONPATH=. pytest tests -q
```

Use `tests/unit`, `tests/contract`, or `tests/integration` for focused runs.
The explicit `PYTHONPATH=.` avoids an environment-specific collision with an
unrelated installed package named `tests`.

## Measure compression ratio

Download a public test NetCDF dataset with:

```bash
.venv/bin/python -m pip install -e ".[analysis]"
mkdir -p data
curl -L --fail --output data/HOAPS_2020-08_6-hourly.nc \
  https://object-store.os-api.cci1.ecmwf.int/esiwacebucket/HOAPS/HOAPS_2020-08_6-hourly.nc
```

Run the benchmark on the `wvpa` variable with:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/HOAPS_2020-08_6-hourly.nc --variable wvpa --bound 0.05

.venv/bin/python scripts/compress_stats.py \
  --input data/HOAPS_2020-08_6-hourly.nc --variable wvpa --sweep 0.01 0.05 0.2
```

The benchmark performs encode and decode, independently checks the bound and
mask, and reports size, CR, RMSE, timing, and container metrics. The current
observed ratios for that sample are approximately `12.50x`, `18.17x`, and
`27.96x` at bounds `0.01`, `0.05`, and `0.2`, respectively. Treat benchmark
values as machine- and build-dependent measurements.

## Project status and acknowledgement

This repository is an experimental research compressor and should be treated
as beta software. File-format and model-version compatibility can change
between releases until a stable `1.0` API is declared.

Development of this project was assisted by large language model (LLM) tools.
The maintainers reviewed, tested, and are responsible for the resulting code,
documentation, and design decisions.

## Documentation map

- [Algorithm](docs/algorithm.md): predictor, quantization, entropy coding,
  repairs, complexity, and guarantees.
- [Architecture](docs/architecture.md): component boundaries and pipeline.
- [Development guide](docs/development.md): installation, Rust build, tests,
  benchmarks, and compatibility discipline.
- [Performance report](docs/performance.md): measured CR and timings.
- [Compression analysis](docs/cr_optimization_analysis.md): why the current
  stencil and entropy modes were selected.

## Repository layout

```text
src/brace_compressor/       Python package and codec implementation
  codec.py                  public API and scan orchestration
  mask.py                   missing-mask extraction and coding
  quant.py                  bound-derived quantization helpers
  model/entropy.py          RAW, RANGE, and context-rANS coding
  verify.py                 bound verification and exact repairs
  container.py              HWPC framing, Zstandard flags, and CRC
rust/brace_scan/            optional bit-exact Rust scan
scripts/                    benchmark and entropy-analysis utilities
tests/                      unit, contract, and integration tests
docs/                       human-facing algorithm and operations docs
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities. This project is
released under the [MIT License](LICENSE).
