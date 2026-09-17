# Algorithm

## Problem and guarantees

The input is a contiguous `float32` or `float64` array with shape `(T, H, W)`. The codec
is lossy for `error_bound > 0`, but guarantees for every valid cell:

```
abs(decoded - original) <= error_bound
```

The missing-value mask is lossless. Missing cells are not predicted or
quantized; their configured sentinel is restored after decoding. Non-finite
values are treated as missing. A zero or sub-epsilon bound uses the same
quantized path as positive bounds, with the configured dtype's machine epsilon
as the floor. It is near-lossless, but can differ by floating-point computation
error. The selected dtype is stored in container metadata and controls
reconstruction, repairs, and output validation.

## 1. Mask extraction

`mask[t, y, x] == True` means that the input cell is missing. The valid
values are flattened in C order, which is also the order used by the
scan and repair positions.

The mask is encoded independently. For structured masks, `mask.py` writes
`HMR1` followed by an initial bit and LEB128 run lengths. For masks where RLE
is larger, it writes NumPy little-endian bitpacking. The decoder detects RLE
from the magic and otherwise expects `ceil(T*H*W/8)` bitpacked bytes.

## 2. Causal prediction

The encoder cannot use an original value to predict a later value because the
decoder will not have that value. Instead, encoding simulates the decoder:

1. Visit cells in `t`, then `y`, then `x` order.
2. Ignore missing cells.
3. Gather already reconstructed neighbors.
4. Compute the weighted mean below, or use the cold-start value `32.0` when
   no usable neighbor exists.
5. Quantize the residual and store the reconstructed value in decoder state.

The longitude-local stencil is:

| Neighbor | Weight | Available when |
| --- | ---: | --- |
| left `(t,y,x-1)` | 8 | `x > 0` and reconstructed value is nonzero |
| top `(t,y-1,x)` | 2 | `y > 0` and reconstructed value is nonzero |
| top-left `(t,y-1,x-1)` | 1 | `y > 0`, `x > 0`, and reconstructed |
| top-right `(t,y-1,x+1)` | 1 | `y > 0`, `x+1 < W`, and reconstructed |
| temporal parent `(t-1,y,x)` | 1 | `t > 0` and reconstructed value is nonzero |

The Rust extension implements the same arithmetic and scan order as the
Python fallback for both float32 and float64. Rust selection is automatic:
there is no codec configuration flag. When `brace_scan` is importable, the
codec dispatches float32 fields to the float32 Rust entry point and float64
fields to the float64 Rust entry point. If the extension or the matching
symbol is unavailable, the codec remains functional through the Python scan.

## 3. Residual quantization

For a positive bound `b`, the step is:

```
Delta = 2 * b
```

For a predicted value `p` and original value `v`, the encoder stores the
signed integer:

```
q = floor((v - p) / Delta + 0.5)
```

The simulated decoded value is `p + q*Delta`. Quantization error is at most
`Delta/2`, i.e. the configured bound apart from floating-point edge cases.
Symbols are clamped to a signed 32-bit-safe range. The container records the
step and origin (`0.0` in the current codec).

## 4. Lossless entropy coding

`model/entropy.py` encodes the `int64` symbols in blocks of 2048. Each block
chooses the smallest lossless representation among:

- `RAW`: fixed-width signed integers using 1, 2, 4, or 8 bytes;
- `RANGE`: zigzag symbols, LEB128 varints, and static byte rANS;
- `CTX`: context-adaptive rANS over the symbol alphabet, using one of three
  frequency tables selected by the previous symbol magnitude (`<=1`, `<=8`,
  or larger).

Zigzag maps `0, -1, 1, -2, 2, ...` to unsigned `0, 1, 2, 3, 4, ...`.
Every entropy mode is bit-exact: decoding reproduces the same symbols. This
is essential because the verifier must assess the exact decoder result.

The entropy payload begins with the block count and mode/width bytes. It then
contains the raw section, range metadata and tables, context metadata and
tables when used, and the repair section. Empty valid-value streams have an
empty residual payload.

## 5. Verification and repair

After the scan, the encoder compares the simulated decoded valid values
with the original valid values. Any position whose error exceeds the bound is
written to a repair map as `(valid_index, exact_value)`, where the value uses
the configured dtype.

The repair map is appended to the residual payload. During decode, repairs are
applied after symbol reconstruction. Thus the returned stream cannot leave a
positive-bound violation unless the configured dtype itself cannot represent
the requested comparison, which is outside the codec's value domain.

## 6. Container and outer compression

The `HWPC` container stores:

1. magic, container version, flags, and scan ABI version;
2. compact JSON metadata;
3. mask payload length and payload;
4. residual payload length and payload;
5. CRC-32 over all preceding bytes.

The optional outer pass is Zstandard and lossless. It is enabled by default by
`BraceCodec`; pass `outer_compress=False` to skip it. The CLI benchmark exposes
the same setting as `--no-outer-compress`. Mask and residual payloads are
considered independently, so incompressible entropy output is not expanded.
Disabling this pass does not disable residual entropy coding or mask packing.
Flag `0x0001` preserves the original meaning that both payloads are compressed;
flags `0x0002` and `0x0004` represent mask-only and residual-only compression.

## Complexity

The scan is `O(T*H*W)` time and stores a reconstructed field of the
same grid footprint. Entropy coding is linear in the number of valid symbols,
with bounded per-block tables. No training, model file, GPU, network, or
runtime data download is required.
