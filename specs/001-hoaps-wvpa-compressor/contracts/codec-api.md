# Contract: Public Codec API (`numcodecs.Codec`)

**Branch**: `remove-transformer` | **Date**: 2026-09-16

Public interface of `hoaps_compressor`. The library exposes a single codec class implementing the `numcodecs.abc.Codec` contract (numcodecs 0.15.0). See [../data-model.md](../data-model.md) for entity details and [../research.md](../research.md) R1/R7 for the numcodecs grounding.

## 1. Class: `HoapsWvpaCodec`

```python
from numcodecs.abc import Codec

class HoapsWvpaCodec(Codec):
    codec_id = "hoaps-wvpa"

    def __init__(self, shape, error_bound, missing_value="nan", dtype="float32"): ...
    def encode(self, buf): ...                 # buffer-like -> bytes (EncodedStream)
    def decode(self, buf, out=None): ...       # bytes (, out) -> buffer-like
    def get_config(self): -> dict              # JSON-serializable, includes "id"
    @classmethod
    def from_config(cls, config): -> HoapsWvpaCodec
```

### Constructor

`__init__(shape, error_bound, missing_value="nan", dtype="float32", outer_compress=True)`

| Parameter | Type | Meaning |
|-----------|------|---------|
| `shape` | `(time, lat, lon)` 3-tuple of positive ints | Grid shape; flat buffers are interpreted as this grid. |
| `error_bound` | finite float ≥ 0 | Absolute error bound (physical wvpa units). **0 is valid** = tightest allowed bound (may still be lossy). Negative or non-finite → `ValueError`. |
| `missing_value` | finite float or `"nan"` | Sentinel marking missing elements. Default `"nan"`. |
| `dtype` | `"float32"` | Fixed in v1; other values → `ValueError`. |
| `outer_compress` | `bool` | Try lossless Zstandard independently on mask and residual payloads. |

### Registration

```python
import numcodecs.registry
numcodecs.registry.register_codec(HoapsWvpaCodec)
```

Performed automatically on `import hoaps_compressor`. After import:

```python
codec = numcodecs.registry.get_codec({"id": "hoaps-wvpa", "shape": (12, 180, 360), "error_bound": 0.01})
```

## 2. Method Contracts

### `encode(buf) -> bytes`

- **Input**: buffer-like of `4·prod(shape)` bytes, interpreted as a float32 grid with the configured `shape`; non-contiguous arrays are copied into a contiguous working buffer. Missing elements equal `missing_value` (or NaN when `missing_value="nan"`).
- **Behavior**: extract mask → predict valid values from causal reconstructed neighbors → quantize residuals with `step = 2 * error_bound` → entropy-code (bit-exact) → verify-and-repair until no reconstructed value deviates by more than `error_bound` (skipped when bound = 0, exempt per FR-003) → frame container with an RLE-or-bitpacked mask, coded residuals, metadata, and checksum.
- **Output**: `bytes` container (self-describing; see §3).
- **Errors**: `ValueError` on wrong buffer size/content; codec never returns a stream that violates the bound (verified internally before return).

### `decode(buf, out=None) -> buffer-like`

- **Input**: container bytes from `encode` (or a byte-compatible stream of the same major container version and model version).
- **`out` semantics** (numcodecs contract): if provided, must be a writeable buffer of exactly `4·prod(shape)` bytes; decoded values are written into it and it is returned. If `None`, a fresh NumPy array (shape `shape`, dtype float32) is returned.
- **Behavior**: validate header/checksum → restore mask bit-exactly → entropy-decode residuals → inverse-quantize with recorded step → replay the deterministic causal predictor → write sentinel into missing positions.
- **Guarantees**: for every valid position, `|decoded − original| ≤ error_bound` (SC-001) when `error_bound > 0`; at `error_bound = 0` the guarantee is exempted (FR-003 exception — output may be lossy). Missing positions identical to original sentinel (SC-002).
- **Errors**: `ValueError` on bad magic/checksum/version, shape/model mismatch, truncated or oversized `out`.

### `get_config() -> dict`

Returns, all JSON-serializable:

```json
{
  "id": "hoaps-wvpa",
  "shape": [12, 180, 360],
  "error_bound": 0.01,
  "missing_value": "nan",
  "dtype": "float32",
  "outer_compress": true
}
```

`missing_value` may be a finite float (e.g. `9.96921e+36`, the NetCDF default fill) or the string `"nan"`.

### `from_config(config) -> HoapsWvpaCodec`

Classmethod; inverse of `get_config`. Accepts the config dict (with `"id"` present; `"id"` validated as `"hoaps-wvpa"`). Round-trip guarantee: `HoapsWvpaCodec(**{k: v for k, v in codec.get_config().items() if k != "id"})` is behaviorally identical to `codec` (same encode/decode byte outputs).

## 3. Encoded Stream / Container Contract

All multi-byte integers little-endian. Fixed layout, length-prefixed payload sections:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic `"HWPC"` |
| 4 | 2 | Container version (uint16, major ABI `1`) |
| 6 | 2 | Flags (`0x0001`: both payloads compressed; `0x0002`: mask compressed; `0x0004`: residual compressed) |
| 8 | 4 | Model version id (uint32) |
| 12 | 4 | Header extra byte length `H` (uint32) |
| 16 | H | Header extra (JSON; shape, dtype, sentinel descriptor, quantization step/lattice, per-block mode count, error_bound as recorded) |
| 16+H | 8 | Mask payload length (uint64) |
| ... | var | Mask payload (RLE or little-endian bitpacked missing mask, stored losslessly) |
| ... | 8 | Residual payload length (uint64) |
| ... | var | Residual payload (per-block mode ids + entropy-coded symbols + repair corrections; empty when no valid values) |
| ... | 4 | CRC-32 checksum over all preceding bytes |

**Compatibility rules**:
- Different major container version → decode MUST fail with a clear error.
- Model version identifies the deterministic causal scan ABI; incompatible versions fail with a clear error.
- `flags` bit0 set → both payload sections are Zstandard-compressed; bits1 and 2 identify mask-only or residual-only Zstandard compression.
- The container independently records everything needed for integrity; `get_config` remains the source of truth for interpretation (numcodecs stores config separately).

## 4. Error Contract (all errors raise `ValueError` unless noted)

| Situation | Error |
|-----------|-------|
| `error_bound` < 0 or non-finite | `ValueError` at construction (FR-008) |
| `shape` not 3 positive ints; `dtype != "float32"` | `ValueError` at construction |
| encode buffer size ≠ `4·prod(shape)` or incompatible dtype | `ValueError` |
| negative/non-finite values in data not matching sentinel | treated as missing (counted and reported via header only; FR-011) |
| decode: bad magic/version/checksum/model mismatch | `ValueError` |
| decode: `out` wrong size | `ValueError` |

## 5. In-Process Verification Hook (FR-012)

Each `encode` computes and includes in the header-extra JSON:

```json
{ "metrics": { "max_abs_error": <float, verified ≤ error_bound>, "n_repaired": <int>, "uncompressed_size": <int>, "payload_size": <int>, "bound_respected": <bool> } }
```

This is verification data (not required for decode correctness) demonstrating SC-001/SC-003 in-process.
