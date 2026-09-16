# Research Notes: HOAPS WVPA Causal Compressor

## Decisions

1. Use the `numcodecs.Codec` contract so the codec can be registered and configured through JSON.
2. Store the missing mask separately and losslessly; never quantize missing positions.
3. Use a deterministic reconstructed-neighbor predictor. Encode simulates decode, so both sides share the same state.
4. Derive quantization from the absolute bound and run a verify-and-repair pass before returning bytes.
5. Keep entropy coding lossless and choose compact payload modes per block.
6. Keep the Rust scan optional; its output must match the Python reference.

These choices provide hard correctness guarantees without runtime model files, training, GPU support, or network access.
