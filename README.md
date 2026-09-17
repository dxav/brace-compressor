# BRACE

[![CI](https://github.com/dxav/brace-compressor/actions/workflows/ci.yml/badge.svg)](https://github.com/dxav/brace-compressor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/optional%20accelerator-Rust-orange.svg?logo=rust)](rust/brace_scan)

**BRACE** (*Bounded Residual Adaptive Compression Engine*) is an error-bounded,
lossless-symbol codec for climate data. It compresses two-dimensional
`(latitude, longitude)` slices or three-dimensional float32/float64 arrays with
shape `(time, latitude, longitude)` using a deterministic reconstructed-neighbor
predictor, bound-derived residual quantization, and lossless entropy coding.
Two-dimensional inputs are represented internally as one time slice.

## Guarantees

- For `error_bound > 0`, every valid decoded value is verified against the
  configured absolute bound before the stream is returned.
- Missing positions are stored in an independent lossless mask and restored
  without loss.
- Entropy coding is bit-exact, so decoded symbols match encoded symbols.
- Rust and Python scans use the same order and arithmetic.
- `error_bound=0` uses the same quantized path as positive bounds, with the
  configured dtype's machine epsilon as the minimum effective error budget.
  It is therefore near-lossless but can differ by floating-point computation
  error.
- `error_bound_mode="relative"` applies the bound pointwise to each nonzero
  value: `abs(decoded - original) <= error_bound * abs(original)`. Exact zeros
  and missing values are preserved separately. The default mode is
  `"absolute"`.
- Supported value dtypes are explicitly selected with `dtype="float32"` or
  `dtype="float64"`; the dtype is recorded in container metadata.

## Codec features

- **Reconstructed-neighbor prediction:** a deterministic causal scan uses
  already reconstructed left, vertical, diagonal, and temporal neighbors, so
  encoding and decoding follow the same predictor state.
- **Bound-derived quantization:** residuals are quantized with a step of
  `2 * max(error_bound, eps(dtype))`, providing a compact integer residual
  stream while preserving the configured absolute-error target through
  verification.
- **Pointwise relative bounds:** relative mode encodes nonzero magnitudes in
  log space with a step of `2 * log1p(error_bound)`, while separate lossless
  masks preserve zero values and signs. This turns the logarithmic error into
  the requested multiplicative bound.
- **Lossless missing-value handling:** missing and non-finite cells are stored
  in an independent RLE or bit-packed mask and restored exactly after decode.
- **Adaptive entropy mode selection:** valid residuals are processed in blocks
  of 2048 symbols. The RANGE candidate selects fixed-width RAW integers or
  static byte rANS over zigzagged LEB128 values independently per block; the
  encoder also compares that payload with a whole-stream context-adaptive
  rANS candidate over the residual-symbol alphabet. RAW stores integers at a
  fixed 1-, 2-, 4-, or 8-byte width; RANGE is the varint-plus-rANS path; and
  CTX is the context-adaptive rANS path.
- **Compact integer coding:** LEB128 stores an integer in 7-bit groups, using
  one continuation bit per byte. Zigzag encoding maps signed residuals so that
  values near zero use the shortest LEB128 representation.
- **Context-adaptive rANS:** the context mode selects one of three frequency
  tables from the previous residual's magnitude: `|residual| <= 1`,
  `|residual| <= 8`, or larger. This models local residual behavior without a
  trained model or external table file.
- **Verify-and-repair:** the encoder simulates the decoder, checks the actual
  reconstructed values, and records exact repairs in the configured dtype for
  any positions that exceed the effective bound.
- **Recommendation-aware strategies:** typed recommendation trees preserve
  `all` as conjunction and `any` as disjunction. For an `any` node, BRACE
  encodes each complete candidate branch, discards candidates whose full
  diagnostics fail, and keeps the smallest valid stream. The selected branch,
  strategy, and diagnostics are stored in the header.
- **Aggregate requirements:** mean absolute, mean relative, and mean
  range-relative requirements use aggregate error budgets. Pointwise children
  in an `all` tree remain active and are intersected with other per-element
  policies.
- **Exact-constraint policies:** data limits and isovalues compile to
  per-element tolerances and exact repairs. Missing-value requirements bind the
  configured sentinel. Lossless requirements use a separate typed-byte path.
- **Self-describing integrity-checked container:** the BRCE format (BRACE
  Container Encoding) records model metadata, payload lengths, compression
  flags, and a CRC-32 checksum. Its four-byte wire magic is `BRCE`, and the
  container version is 3.
- **Optional Zstandard pass:** mask and residual payloads can be compressed
  independently with Zstandard when that reduces their size; this pass is
  lossless and does not replace the residual entropy modes. It is enabled by
  default and can be disabled with `BraceCodec(..., outer_compress=False)` or
  the benchmark CLI's `--no-outer-compress` option. Disabling it skips only
  this final Zstandard pass; entropy coding and lossless mask coding remain
  enabled.
- **Optional Rust acceleration:** the causal scan and RANGE/CTX rANS
  primitives have bit-exact Rust implementations with Python fallbacks. RAW
  packing, entropy-mode selection, and dispatch remain in Python. There is no
  codec configuration flag: when the compiled `brace_scan` extension is
  importable, the codec automatically uses each matching Rust implementation.
  If the extension or a matching symbol is unavailable, the corresponding
  Python implementation is used instead.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

The package is built with `maturin` and can include the optional Rust
accelerator. Installing the package with a built extension activates Rust
scanning automatically; no additional runtime setting is required. The codec
still works without a Rust toolchain by using the Python scan.

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

For a pointwise relative bound, select the relative mode explicitly:

```python
codec = BraceCodec(
  shape=shape,
  error_bound=0.01,
  error_bound_mode="relative",
)
```

Recommendations from the typed `compression-recommendations` package can be
applied directly. Pointwise-relative bounds are preferred when available;
otherwise the adapter selects a pointwise absolute bound. Mean absolute and
mean relative bounds use aggregate error budgets with exact repairs for the
largest residual contributors. `any` branches are selected against the input
data's scale at encode time and the selected branch is recorded in metadata.
Level-specific searches can pass
`level_kind`, such as `"single"` or `"pressure"`.

```python
from brace_compressor import BraceCodec

codec = BraceCodec.from_recommendation(
  shape=shape,
  variable="cc",
  dtype="float32",
)
```

`recommend_error_bound("cc")` returns the static recommendation mode and value
as an `ErrorBoundRecommendation`. The codec factory retains the complete
typed plan. At encode time, range-relative bounds are resolved from the input
range, quadratic bounds may use deterministic block-local quantization steps,
and `any` alternatives are compared by actual encoded size. Data limits and
isovalues tighten per-element tolerances, missing-value recommendations bind
the exact sentinel mask, and lossless recommendations use a byte-shuffled
typed-byte stream that preserves signed zero and NaN payloads.

The encoded header contains `recommendation_plan` and
`recommendation_checks`. Each check includes its kind, pass/fail result,
metric, limit, violation count, and nested child checks. The benchmark treats
these full diagnostics as authoritative; a static scalar recommendation is
reported separately when it does not describe the selected branch.

For recommendation plans that only specify exact constraints, provide an
explicit `error_bound` to control the lossy base strategy for values not
covered by those constraints:

```python
codec = BraceCodec.from_recommendation(
  shape=shape,
  variable="x",
  error_bound=0.1,
  recommendations=recommendations,
)
```

### ERA5 recommendation example

The repository includes a complete example for the downloaded ERA5
pressure-level challenge dataset. It loads each variable, extracts its typed
recommendation, compresses and decompresses it, checks the selected bound, and
writes JSON results:

```bash
PYTHONPATH=src python scripts/compress_era5_with_recommendations.py \
  --input data/era5_pressure_20260715T1200_4levels.nc \
  --output data/era5_pressure_recommendation_example.json
```

Use `--variable cc --variable t` to run a smaller example. The package's
pressure-level recommendations are selected from the variable's CF/GRIB short
name; for example, `cc` uses a 1% pointwise-relative bound while `t` uses a
0.05 pointwise-absolute bound. The output JSON includes the selected
requirement tree, strategy, repair count, and full requirement diagnostics.

The current repository input is
`data/era5_pressure_20260715T1200_4levels.nc`. The recommendation benchmark
fails if any selected requirement tree fails after decoding and reports the
aggregate compression ratio.

To disable the final lossless Zstandard pass explicitly:

```python
codec = BraceCodec(
  shape=shape,
  error_bound=0.05,
  outer_compress=False,
)
```

`BraceCodec` inherits from `numcodecs.abc.Codec` and implements `codec_id`,
`encode`, `decode`, `get_config`, and `from_config`. Importing
`brace_compressor` registers the `brace` codec with the registry.

Use `dtype="float64"` to preserve float64 input precision and decode into a
float64 output buffer. Existing float32 streams remain the default format.

## Run tests

```bash
PYTHONPATH=src pytest tests -q
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

Add `--no-outer-compress` to benchmark without the final Zstandard pass:

```bash
.venv/bin/python scripts/compress_stats.py \
  --input data/HOAPS_2020-08_6-hourly.nc \
  --variable wvpa --bound 0.05 --no-outer-compress
```

The benchmark performs encode and decode, independently checks the bound and
mask, and reports size, CR, RMSE, timing, and container metrics. In the
current reference run, the ratios were `12.25x`, `17.78x`, and `27.60x` at
bounds `0.01`, `0.05`, and `0.2`, respectively. See the
[performance report](docs/performance.md) for the dataset, timings, and caveats;
benchmark values are machine- and build-dependent measurements.

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
  container.py              BRCE framing, Zstandard flags, and CRC
rust/brace_scan/            optional bit-exact Rust scan
scripts/                    benchmark and entropy-analysis utilities
tests/                      unit, contract, and integration tests
docs/                       human-facing algorithm and operations docs
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities. This project is
released under the [MIT License](LICENSE).
