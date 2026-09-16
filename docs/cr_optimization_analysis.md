# Compression-Ratio Optimization Analysis

The codec's compression ratio is controlled by three independent choices:

1. The legal lattice step is `Delta = 2 * error_bound`, giving quantization
   error no greater than the requested positive bound.
2. HOAPS water-vapor data has stronger spatial than temporal correlation, so
   the causal stencil uses weights `(left 5, top 5, top-left 2, top-right 2,
   temporal 1)`.
3. Context-adaptive lossless entropy coding exploits conditional structure in
   residual symbols without changing decoded values.

The encoder and decoder use the same deterministic causal state. The encoder
simulates the decoder and the verifier emits exact repairs for any outlier.
The optional Rust scan and Python reference produce byte-identical streams.
