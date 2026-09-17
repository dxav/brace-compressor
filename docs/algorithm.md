# Algorithm

## Problem and guarantees

The input is a contiguous `float32` or `float64` array with shape `(T, H, W)`.
The public API also accepts `(H, W)`, represented internally as one time
slice. In absolute mode, the codec is lossy for `error_bound > 0`, but
guarantees for every valid cell:

```
abs(decoded - original) <= error_bound
```

The missing-value mask is lossless. Missing cells are not predicted or
quantized; their configured sentinel is restored after decoding. Non-finite
values are treated as missing. A zero or sub-epsilon bound uses the same
quantized path as positive bounds, with the configured dtype's machine epsilon
as the floor. It is near-lossless, but can differ by floating-point computation
error. The selected dtype is stored in container metadata and controls
reconstruction, repairs, and output validation. Any non-finite input is
treated as missing.

`error_bound_mode="absolute"` is the default and applies one absolute bound
to every valid value. With `error_bound_mode="relative"`, every nonzero valid
value is required to satisfy:

```
abs(decoded - original) <= error_bound * abs(original)
```

Relative mode uses the dtype epsilon as the floor for a requested zero bound.
Zeros are represented exactly, and the sign of every nonzero value is stored
losslessly. Relative mode is encoded in log-magnitude space and uses additional
zero and sign mask sections in the container.

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

For a requested bound `b` and configured dtype `d`, the effective bound is
`max(b, eps(d))`. The step is:

```
Delta = 2 * max(b, eps(d))
```

For a predicted value `p` and original value `v`, the encoder stores the
signed integer:

```
s = floor((v - p) / Delta + 0.5)
symbol = s - 2**31
```

The simulated decoded value is `p + s*Delta + origin`, with `origin = 0.0` in
the current codec. Symbols are clamped to a signed 32-bit-safe range before the
offset is applied. The decoder reverses the offset, so the quantization error
is at most `Delta/2`, apart from floating-point edge cases.

For relative mode, the scan input is `log(abs(v))` plus a fixed offset, and the
step is:

```
Delta_relative = 2 * log1p(max(error_bound, eps(d)))
```

The decoded magnitude is `exp(decoded_log - offset)`. Half a step in log space
is `log1p(relative_bound)`, so the multiplicative error is bounded by the
requested relative bound apart from floating-point edge cases. Exact zeros and
signs are restored from the additional lossless masks.

## 4. Lossless entropy coding

`model/entropy.py` encodes the `int64` symbols in blocks of 2048. It builds two
lossless candidates and emits the smaller one:

- **RAW**: fixed-width signed integers using 1, 2, 4, or 8 bytes. This is
  useful when a block is cheaper to store directly than to entropy-code.
- **RANGE**: signed residuals are first zigzag-encoded, then written as
  LEB128 varints, and finally compressed as bytes with a static rANS model.
  RANGE is selected independently per 2048-symbol block against fixed-width
  RAW coding.
- **CTX**: the complete stream is encoded with context-adaptive rANS over the
  residual-symbol alphabet, using one of three frequency tables selected by
  the previous symbol magnitude (`<=1`, `<=8`, or larger). CTX is skipped when
  its symbol span would make the tables too large.

LEB128 is a variable-length integer encoding: each byte contributes seven data
bits, while its high bit says whether another byte follows. Zigzag encoding
maps signed values near zero to small unsigned values (`0, -1, 1, -2, 2` maps
to `0, 1, 2, 3, 4`), making those values compact under LEB128.

rANS means **range Asymmetric Numeral Systems**. It is a table-based entropy
coder that represents frequent symbols with fewer bits than rare symbols. The
RANGE path uses one static byte-frequency table. The CTX path selects one of
three symbol-frequency tables from the preceding residual's magnitude. The
Rust and Python implementations use the same rANS scale and normalized tables,
so the resulting payload is bit-exact across implementations.

When the compiled `brace_scan` extension is available, both rANS variants are
implemented in Rust and selected automatically by `model/entropy.py`. The
Python implementations remain the reference fallback, so installing the
extension changes execution time but not the payload format or bitstream.

Zigzag maps `0, -1, 1, -2, 2, ...` to unsigned `0, 1, 2, 3, 4, ...`.
Every entropy mode is bit-exact: decoding reproduces the same symbols. This
is essential because the verifier must assess the exact decoder result.

The entropy payload begins with the block count and mode/width bytes. It then
contains the raw section and the selected RANGE or CTX metadata, frequency
tables, and rANS stream. The codec's residual payload wraps that entropy payload
with an 8-byte entropy-payload length and then appends the repair section. A
stream with no valid values has an empty entropy payload and still carries the
repair-section framing.

## 5. Verification and repair

After the scan, the encoder compares the simulated decoded valid values
with the original valid values. Any position whose error exceeds the bound is
written to a repair map as `(valid_index, exact_value)`, where the value uses
the configured dtype.

The repair map is appended to the residual payload as a count followed by
int64 valid-value positions and exact values in the configured dtype. During
decode, repairs are applied after symbol reconstruction. Thus the returned
stream cannot leave a positive-bound violation unless the configured dtype
itself cannot represent the requested comparison, which is outside the codec's
value domain.

## 6. Container and outer compression

The **BRCE** format, short for **BRACE Container Encoding**, stores:

1. magic, container version, flags, and scan model version;
2. compact JSON metadata;
3. mask payload length and payload;
4. residual payload length and payload;
5. CRC-32 over all preceding bytes.

The on-disk magic is the four-byte ASCII value `BRCE`. Container version 3
identifies this format.

The optional outer pass is Zstandard at level 3 and is lossless. It is enabled
by default by `BraceCodec`; pass `outer_compress=False` to skip it. Mask and
residual payloads are considered independently, so incompressible entropy
output is not expanded. Disabling this pass does not disable residual entropy
coding or mask packing. Flag `0x0001` means both payloads are compressed;
flags `0x0002` and `0x0004` represent mask-only and residual-only compression.
The container ends with CRC-32 over all preceding bytes, and BRCE v3 rejects
other container versions.

## Complexity

The scan is `O(T*H*W)` time and stores a reconstructed field of the
same grid footprint. Entropy coding is linear in the number of valid symbols,
with bounded per-block tables. No training, model file, GPU, network, or
runtime data download is required.
