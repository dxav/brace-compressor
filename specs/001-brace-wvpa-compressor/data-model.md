# Data Model: BRACE WVPA Codec

## Codec

`BraceCodec` stores `shape`, `error_bound`, `missing_value`, `dtype`, and `outer_compress`. The only supported dtype is float32. Config values are JSON serializable.

## Missing mask

A boolean array with `True` meaning missing. It is bitpacked or RLE-coded as an independent lossless payload. Missing positions are restored with the configured sentinel.

## Quantized residuals

Signed integer symbols represent residuals from the deterministic reconstructed-neighbor predictor on a bound-derived lattice. The entropy payload reproduces these symbols exactly.

## Causal predictor

The scan uses reconstructed left, top, top-left, top-right, and temporal-parent values with weights 8, 2, 1, 1, and 1. Cold starts use `32.0`. The decoder repeats the same scan.

## Encoded stream

The container contains magic/version/flags, a scan ABI version, JSON metadata, mask payload, residual payload, optional lossless Zstandard compression, and CRC-32.
