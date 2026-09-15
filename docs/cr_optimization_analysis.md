# Compression-Ratio Optimization Analysis

**Date**: 2026-09-15  
**Target**: hard-error-bounded compression of the HOAPS `wvpa` field

This document records how the causal predictor was improved and how the final
weights were selected. The experiments deliberately separate quantization
effects, predictor quality, and entropy-coder behavior. That separation is
important because the largest observed CR increase was not caused by learned
weights.

## 1. Establish a controlled baseline

All predictor comparisons used the same real HOAPS field, scan order, missing
mask, codec framing, and entropy coder. The primary field was
`wvpa_2020-08-01_07.npy`, with shape `(28, 320, 720)` and 68.7% missing cells.
The bound was fixed at `0.05` unless stated otherwise.

The quantization step was first corrected to

$$
\Delta = 2 \times \text{error\_bound}.
$$

A residual is rounded to this lattice and therefore has quantization error at
most `Delta / 2`, which is the requested absolute bound. The predictor is
computed identically by the encoder and decoder, so it does not add a second
reconstruction error term. Verification and repair still check the complete
round trip.

This correction changed the real-data result from 11.67x to 16.34x at bound
`0.05`. It also changed the maximum quantization error from `0.0125` to
`0.05`. The increase must therefore be attributed to the larger legal
quantization step, not to transformer training or causal weights.

As a control, random and trained transformer priors were compared at the same
step. Both produced 16.34x at bound `0.05` and 11.07x at bound `0.01`.
Training reduced the coarse-grid loss slightly, but did not improve CR. It
also made cold-start prediction slightly worse:

| Prior | Cold-start MAE | Cold-start RMSE |
|---|---:|---:|
| Random | 15.70 | 18.56 |
| Trained | 16.23 | 20.30 |

The prior only affects cold-start cells, approximately 10 thousand of 2.02
million valid cells in this field. The causal scan dominates the residual
entropy, so the trained prior was not selected as the source of the CR gain.

## 2. Measure where the predictability is

The causal predictor can use only values already reconstructed by both sides.
The tested neighbors were:

- temporal parent: the same spatial cell at the previous time step;
- left: the previously scanned longitude cell;
- top: the previously scanned latitude cell;
- top-left and top-right: causal diagonal spatial neighbors.

For each candidate source, the original field was used only for analysis to
measure prediction error. Original values were never used by the shipped
codec, because they are unavailable to the decoder.

The neighbor measurements showed that spatial correlation is substantially
stronger than temporal correlation:

| Candidate source | Approx. MAE |
|---|---:|
| Temporal parent | 2.049 |
| Left neighbor | 0.797 |

This falsified the assumption that the temporal parent should receive most of
the weight. The field changes more slowly across nearby spatial cells than it
does between the sampled time slices, so the next experiment shifted weight
toward the spatial stencil.

## 3. Search candidate causal stencils

The predictor is a normalized weighted average of the available causal
neighbors. Candidate sets were evaluated using residual symbol entropy and
then checked with the complete codec. The old stencil was:

```text
temporal=4, left=2, top=2, top-left=1, top-right=1
```

The selected spatial-weighted stencil was:

```text
temporal=1, left=5, top=5, top-left=2, top-right=2
```

The representative residual entropy measurements were:

| Stencil | Entropy |
|---|---:|
| Old temporal-weighted `(4, 2, 2, 1, 1)` | 5.704 bits/symbol |
| Selected spatial-weighted `(1, 5, 5, 2, 2)` | 5.44 bits/symbol |

The selected weights reduced entropy without adding state that the decoder
could not reproduce. A perfect predictor using the original neighbors was
also tested as an upper-bound experiment. It provided only limited additional
headroom, indicating that the fixed spatial stencil is near the practical
ceiling for this causal scan and field.

The choice is therefore called **near optimal**, not mathematically optimal:
the tested candidates and the perfect-neighbor upper bound show that large
additional gains are unlikely from small weight adjustments, but they do not
prove a global optimum over every possible predictor.

## 4. Account for the entropy coder

The entropy result was evaluated at the symbol level as well as at the final
byte-stream level. Residuals are zigzag/varint encoded and then stored in
per-block entropy payloads. The coder can choose a compressed rANS payload or
fall back to a RAW payload when compression overhead would be larger.

This check matters because lower residual entropy does not automatically imply
an identical CR gain: block headers, varint lengths, rANS tables, RAW blocks,
mask storage, repairs, and container framing also contribute to the final
size. The final decision was therefore based on the complete encoded stream,
not entropy alone.

The spatial-weighted stencil reduced the residual entropy and produced the
following end-to-end result on the real field:

| Predictor | Compression ratio | Maximum error |
|---|---:|---:|
| Old temporal-weighted | 16.34x | 0.0500 <= 0.05 |
| New spatial-weighted | **17.03x** | 0.0500 <= 0.05 |

## 5. Exploit the conditional entropy (T048)

The entropy payload after the stencil change was 1,638,635 bytes, against an
ideal 1,373,684 bytes at the measured 5.44 bits/symbol unconditional entropy.
Two observations drove the next step:

1. **The per-block mode heuristic was systematically wrong.** It compared
   the *varint* size against the raw packed size, but rANS on the varint
   bytes is almost always smaller than raw packing. Re-running the decision
   with the actual rANS size flipped all 361 RAW blocks to RANGE, saving
   258,862 bytes (15.8%) of the entropy payload. The outer zlib pass had
   been masking part of this gain, but the rANS-first decision is strictly
   better.

2. **The symbols have strong conditional structure.** The first-order
   conditional entropy is 4.84 bits/symbol vs 5.44 unconditional — an 11%
   reduction. The residual distribution depends on the previous residual's
   magnitude: smooth regions produce tiny residuals, edges produce larger
   ones.

A **context-adaptive rANS** mode (`MODE_CTX`) was added: each symbol is
coded with a probability table selected by the previous symbol's magnitude
bucket (small/medium/large). The encoder builds both the RANGE and CTX
payloads and emits the smaller; a cheap entropy-based pre-decision skips
the RANGE build when CTX is clearly better. The CTX path is bit-exact and
preserves the hard error bound.

Measured on the real field:

| Bound | Old CR (RANGE) | New CR (CTX) | Max error |
|-------|---------------:|-------------:|----------:|
| 0.01  | 11.07x | **12.45x** | 0.0100 <= 0.01 |
| 0.02  | 13.70x | **14.34x** | 0.0200 <= 0.02 |
| 0.05  | 17.03x | **17.91x** | 0.0500 <= 0.05 |
| 0.1   | 20.37x | **21.74x** | 0.1000 <= 0.1 |

The entropy payload dropped to 1,321,172 bytes (from 1,638,635), and the
rANS stream is now within ~0.01 bits/symbol of the conditional entropy
floor. The remaining headroom is in reducing the symbol entropy itself
(better prediction), not the coder.

## 6. Implement and validate both scan paths

The selected weights were implemented in both causal scan implementations:

- the Python reference path in `src/hoaps_compressor/codec.py`;
- the Rust/PyO3 accelerated path in `rust/hoaps_scan/src/lib.rs`.

The Rust path must use the same scan order, neighbor availability rules,
weight normalization, quantization, and decoder-state update as Python. This
was validated by comparing the encoded streams directly: Python and Rust
produce byte-identical output.

The complete validation also checked:

- maximum absolute reconstruction error never exceeds the configured bound;
- the missing-value bitmask is preserved exactly;
- encode/decode is deterministic;
- Python and Rust streams are byte-identical;
- the real-field CR is 17.91x at bound `0.05`;
- the full test suite passes (`77 passed`).

## Conclusion

The analysis found three different improvements with different causes:

1. Changing the legal lattice step to `Delta = 2 * bound` explains the major
   11.67x to 16.34x increase.
2. Measuring residual structure and shifting the stencil toward spatial
   neighbors explains the further 16.34x to 17.03x increase.
3. Exploiting the conditional entropy of the residual symbols with a
   context-adaptive rANS explains the further 17.03x to 17.91x increase.

The trained transformer prior did not improve CR and remains useful only as a
deterministic cold-start prior. The final spatial-weighted causal predictor is
small, reproducible, hard-error compatible, and close to the practical limit
suggested by the original-neighbor upper-bound experiment. The entropy coder
is now within ~0.01 bits/symbol of the conditional entropy floor, so further
CR gains must come from better prediction.
