# BRACE

[![CI](https://github.com/esiwace/brace-compressor/actions/workflows/ci.yml/badge.svg)](https://github.com/esiwace/brace-compressor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/optional%20accelerator-Rust-orange.svg?logo=rust)](rust/brace_scan)

**BRACE** (*Bounded Residual Adaptive Compression Engine*) is an error-bounded,
lossless-symbol `numcodecs` codec for
HOAPS water-vapor (`wvpa`) climate fields. It compresses two-dimensional
`(latitude, longitude)` slices or three-dimensional float32 arrays with shape
`(time, latitude, longitude)` using a deterministic reconstructed-neighbor
predictor, bound-derived residual quantization,
and lossless entropy coding.

An optional Rust extension accelerates
the scan; the Python implementation remains the fallback.

## Guarantees

- For `error_bound > 0`, every valid decoded value is verified against the
  configured absolute bound before the stream is returned.
- Missing positions are stored in an independent lossless mask and restored
  with the configured sentinel.
- Entropy coding is bit-exact, so decoded symbols match encoded symbols.
- Rust and Python scans use the same order and arithmetic.
- `error_bound=0` is accepted and uses the tightest available float32 path;
  the positive-bound guarantee is explicitly exempt for this mode.

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

For the compression-lab missing-values challenge, a selected 2-D xarray slice
can be passed directly using the same `numcodecs` interface:

```python
field = da.values.astype("float32", copy=False)
codec = BraceCodec(shape=field.shape, error_bound=1.0, missing_value="nan")
encoded = codec.encode(field)
decoded = codec.decode(encoded, out=np.empty(field.shape, dtype="float32"))

assert np.array_equal(np.isfinite(decoded), np.isfinite(field))
assert np.max(np.abs(decoded[np.isfinite(field)] - field[np.isfinite(field)])) <= 1.0
compression_ratio = field.nbytes / np.asarray(encoded).nbytes
```

Two-dimensional inputs are encoded internally as a single time slice, while
the output buffer keeps its original 2-D shape.

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

The bundled real sample is a `(28, 320, 720)` HOAPS-like field:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/wvpa_2020-08-01_07.npy --bound 0.05

.venv/bin/python scripts/compress_stats.py \
  --input data/wvpa_2020-08-01_07.npy --sweep 0.01 0.05 0.2
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
- [Container/API contract](specs/001-brace-wvpa-compressor/contracts/codec-api.md):
  public API and binary framing.
- [Development guide](docs/development.md): installation, Rust build, tests,
  benchmarks, and compatibility discipline.
- [Performance report](docs/performance.md): measured CR and timings.
- [Data model](specs/001-brace-wvpa-compressor/data-model.md): stream and
  intermediate entities.
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
specs/                      stable API/data-model requirements
```

Large source datasets, local benchmark reports, virtual environments, build
outputs, and Python caches are intentionally excluded from version control.
The included notebook uses the challenge S3 dataset through
`open_remote_dataset` when data access is available.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities. This project is
released under the [MIT License](LICENSE).
