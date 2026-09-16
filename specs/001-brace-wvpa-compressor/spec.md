# Feature Specification: BRACE WVPA Codec

**Feature Branch**: `remove-transformer`

The compressor reduces HOAPS water-vapor (`wvpa`) arrays through deterministic reconstructed-neighbor prediction, bound-tied quantization, and lossless entropy coding.

## Requirements

- Accept a contiguous float32 `(time, latitude, longitude)` array through the `numcodecs.Codec` API.
- Preserve missing-value locations exactly using a separately stored compact mask.
- For every finite absolute error bound, guarantee decoded valid values are within the effective bound `max(error_bound, float32_epsilon)`; reject negative and non-finite bounds. Zero is accepted as a near-lossless epsilon-clamped mode.
- Predict each valid cell only from the missing mask, a deterministic cold-start value, and already reconstructed spatial/temporal neighbors.
- Exploit both spatial and temporal correlation and retain a Python reference scan plus an optional bit-exact Rust scan.
- Encode residual symbols losslessly, frame payloads with versioned metadata and CRC, and report compression metrics.
- Handle all-missing, no-missing, and non-finite input cases according to the codec contract.

## Success Criteria

- Round trips preserve the missing mask bit-for-bit.
- Positive-bound round trips pass the error-bound verifier with zero violations.
- The encoded stream is smaller than the source for typical smooth HOAPS-like fields.
- Config round trips through `get_config` and `from_config` remain JSON serializable and behaviorally identical.
- Space-time prediction is at least as compact as the per-field baseline within the integration-test tolerance.
