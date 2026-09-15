# Implementation Plan: HOAPS WVPA Transformer Compressor

**Branch**: `001-hoaps-wvpa-attention` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-hoaps-wvpa-compressor/spec.md`

## Summary

Build a Python library exposing a `numcodecs.Codec`-compatible compressor for the HOAPS water vapor (wvpa) variable. The codec exploits both spatial and temporal structure with a transformer-based model (optionally using JPEG AI-style learned compression techniques and attention mechanisms) and guarantees a hard, user-specified absolute error bound: every reconstructed value differs from the original by at most the bound, except at bound zero where the guarantee is explicitly exempted (bound 0 = tightest allowed bound, may still be lossy). Missing values are preserved exactly via a compact, losslessly stored bitmask. The error bound is never violated (for bound > 0); precision is adaptively increased where the data is hard to approximate, and the design maximizes compression ratio at any given bound. The transformer model is introduced in User Story 4; earlier stories may use simpler placeholder predictors.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: numcodecs (>=0.12, Codec ABC contract), numpy, PyTorch (transformer model and learned entropy coding), pytest (testing)

**Storage**: N/A (library; compressed bytes are returned to the caller or consumed from a buffer). Encoded stream is a self-describing byte container (see contracts/codec-api.md).

**Testing**: pytest with property/round-trip tests over generated gridded fields

**Target Platform**: Linux (single machine, CPU; GPU optional acceleration, not required)

**Performance Goals**: Compression and decompression of a monthly HOAPS wvpa gridded time series (≈ 360 × 180 × N time steps, float32 ≈ 1–50 MB) completes in minutes on a single machine (offline batch acceptable); compression ratio ≥ 2× (≥ 50% smaller) at a reasonable error bound (SC-003), and ≥ spatial-only baseline at any given bound (SC-007).

**Constraints**: Hard absolute error bound never violated on any value (FR-003/FR-016), except when the bound is zero — bound=0 is explicitly exempted from the FR-003 guarantee (FR-017) and may be lossy; missing-value mask preserved bit-exactly (FR-004/FR-015); codec parameters JSON-serializable per `numcodecs.Codec.get_config()`; single-machine, offline; no real-time requirement.

**Scale/Scope**: Single variable (HOAPS wvpa) compression; gridded space-time fields; v1 scope excludes CLI, real-time/streaming, and distributed processing.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is an un-ratified template containing no binding principles or gates. **Result: no ratified constitution — no gates apply.** Re-checked after Phase 1 design (single library project, standard test layout): no violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-hoaps-wvpa-compressor/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   └── codec-api.md     # numcodecs.Codec public API + encoded byte-container contract
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
# Single project (library)
src/hoaps_compressor/
├── __init__.py
├── codec.py             # HoapsWvpaCodec(numcodecs.abc.Codec): encode/decode/get_config/from_config/codec_id
├── container.py         # Encoded byte-container read/write: header, bitmask, residual payloads, metadata
├── mask.py              # Missing-value mask extraction, compact bitpacking, restoration
├── bound.py             # Error-bound validation (reject negative/non-finite; zero = tightest bound)
├── model/
│   ├── __init__.py
│   ├── transformer.py   # Space-time transformer predictor (attention-based, JPEG AI-inspired)
│   └── entropy.py       # Learned/entropy coding of quantized residuals
├── quant.py             # Residual quantization bound-tied to the absolute error bound
└── verify.py            # Post-encode verification: max |error| ≤ bound, mask identical

tests/
├── contract/            # numcodecs.Codec contract tests (encode/decode/get_config/from_config/codec_id)
├── integration/         # End-to-end round trips on HOAPS wvpa-like gridded fields
└── unit/                # mask, bound, container, quantization unit tests
```

**Structure Decision**: Single library project under `src/hoaps_compressor/` with a `tests/` tree split into contract/integration/unit, matching the `numcodecs.Codec` drop-in library delivery chosen in the spec (FR-014). No CLI, no web/mobile tiers.

## Complexity Tracking

> No Constitution Check violations to justify (constitution is an un-ratified template; no gates apply).

---

## Phase 8 (Enhancement): Intensive Transformer & Attention Prediction

**Branch**: `001-hoaps-wvpa-attention` | **Date**: 2026-09-15

**Goal**: Increase compression ratio by making the transformer/attention the primary predictor during the causal scan, not just a cold-start prior. The current codec calls `TransformerPredictor.base_prior(mask)` once before scanning; after the first reconstructed neighbor exists, prediction is a fixed weighted average of causal neighbors (temporal parent 4, left 2, top 2, top-left 1, top-right 1). This leaves most residual entropy untouched by attention.

### Problem statement

- `base_prior` runs once per encode/decode on the mask only. It is a smooth cold-start hint.
- The causal scan (Rust/Python) does the real per-cell prediction from reconstructed neighbors.
- Deeper global attention or a larger token budget would mostly refine cold starts, not the dominant residual stream.
- Full-grid attention over a 6.45 M-cell HOAPS field is O(N²) and infeasible.

### Design direction

1. **Train the existing transformer** on HOAPS masked-cell prediction (self-supervised: sample a mask, predict hidden valid values, MSE/Huber loss). Export deterministic, versioned weights via the existing `serialize_weights()`/`load_weights()` mechanism; bump `MODEL_VERSION`.
2. **Use attention during the causal scan** — process valid cells in causal blocks (e.g. 128–256 positions). For each block, feed the transformer:
   - reconstructed temporal neighbors,
   - reconstructed spatial neighbors,
   - the mask,
   - positional encodings,
   with a **causal/local attention mask** so each position attends only to already-reconstructed cells. The transformer output becomes the prediction; the existing weighted predictor remains a fallback.
3. **Train with decoder-state simulation** — feed reconstructed/quantized previous values (not original values) so the model does not rely on teacher forcing that disappears at decode.
4. **Keep the hard error guarantee unchanged** — the transformer only predicts the residual center; quantization stays lossless at the symbol level; `verify_and_repair()` remains the final correctness boundary.

### Implementation order

1. Fix and test the `out_scale` initialization (currently zeroed, so the prior is near-constant).
2. Add trained-weight loading for a HOAPS model.
3. Benchmark the trained base prior alone.
4. Add block-local causal attention in Python first (correctness), then port the finalized block predictor to Rust/TorchScript.
5. Compare residual entropy, repair count, runtime, and compression ratio.

### Success criteria (this phase)

- Compression ratio strictly higher than the current baseline at the same error bound on `data/wvpa_2020-08-01_07.npy` (SC-008).
- Encode/decode remain bit-identical (deterministic predictor, identical causal walk).
- Hard error bound and mask preservation guarantees unchanged (all existing tests still pass).

## Complexity Tracking (Enhancement)

> No Constitution Check violations to justify (constitution is an un-ratified template; no gates apply).
