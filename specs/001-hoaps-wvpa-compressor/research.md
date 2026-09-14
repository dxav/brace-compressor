# Research: HOAPS WVPA Transformer Compressor

**Branch**: `001-hoaps-wvpa-compressor` | **Date**: 2026-09-14

This document resolves the technical unknowns from the plan's Technical Context. Each entry records the decision, rationale, and alternatives considered.

## R1. Codec API surface: which `numcodecs.Codec` methods must be implemented?

**Decision**: Implement the full `numcodecs.abc.Codec` contract of numcodecs 0.15.0:
- `codec_id` class attribute (string, e.g. `"hoaps-wvpa"`),
- `encode(buf)` → encoded buffer,
- `decode(buf, out=None)` → decoded buffer (optional `out` must be honored when provided),
- `get_config()` → dict of JSON-serializable config incl. `"id"` field,
- `from_config(config)` classmethod returning an instance.

**Rationale**: The spec (FR-014) mandates drop-in `numcodecs.Codec` compatibility. The docs define exactly these members; `get_config` values must be JSON-serializable and include `id`, and two classes share a `codec_id` only if fully compatible. Registering via `numcodecs.registry.register_codec(HoapsWvpaCodec)` enables `get_codec({"id": "hoaps-wvpa", ...})` and Zarr integration.

**Alternatives considered**:
- A bespoke API (compress/decompress functions) — rejected: violates FR-014's drop-in requirement.
- Zarr v3-specific codec (numcodecs.zarr3) — rejected for v1: narrows compatibility; the base `Codec` ABC is the required interface. A Zarr 3 adapter can be added later without contract change.

## R2. Input data model: what does `encode(buf)` receive?

**Decision**: `encode` accepts a buffer exporting a contiguous, C-contiguous `numpy.ndarray`-viewable memory region. The codec interprets the buffer as a space-time gridded field of float32 values.

**Rationale**: numcodecs codecs operate on buffer protocols, not `xarray`/`NetCDF` objects. HOAPS wvpa distributed as NetCDF is a gridded space-time array; a contiguous buffer is the numcodecs-idiomatic representation. The array is interpreted by configuration (grid shape, missing sentinel), as HOAPS wvpa is a 3D (time, lat, lon) gridded field.

**Alternatives considered**:
- Accept netCDF4/xarray Dataset objects — rejected: breaks `Codec` buffer contract and forces an xarray dependency into the codec core.
- Accept arbitrary N-D arrays without configuration — rejected: the missing-value sentinel and grid interpretation must come from config (`get_config`), per the numcodecs design that parameters live outside the stream.

## R3. Missing-value handling and mask storage

**Decision**: On encode, classify elements against the configured missing sentinel (e.g. NaN or a sentinel float), extract a **missing-value bitmask** (1 bit per element, bitpacked), and compress only valid values. The bitmask is stored **separately** as its own payload in the encoded container, unmodified (optionally with a general-purpose post-pass, which is always lossless). On decode, the mask is restored bit-exactly and missing positions are re-materialized with the sentinel; valid positions are filled with reconstructed values.

**Rationale**: FR-004/FR-015 require bit-exact mask preservation, and the clarification (Session 2026-09-14) directs storing the mask separately as a compact bitmask to maximize CR. Separation also lets the transformer operate exclusively on valid values (no learning across the sentinel), which improves both prediction and bound enforcement.
- All-missing input → no value payload at all (maximal compression); empty input → zero-size field handled by header metadata.

**Alternatives considered**:
- Embedded sentinel-aware model input (mask inside the transformer) — rejected for v1: complicates bound verification; can be revisited as an optimization.
- Undefined behavior for stray NaNs — rejected: FR-011 requires documented behavior; stray non-finite values that don't match the sentinel are treated as missing and counted/reported (documented in contracts).

## R4. Error-bound enforcement mechanism (never violate; maximize CR)

**Decision**: **Predict → quantize residual → entropy-code → verify-and-repair.**
1. A space-time transformer predicts each valid value from context (JPEG AI-style learned transform with attention).
2. Residuals are quantized with a quantization step derived from the absolute error bound (step ≤ bound), and entropy-coded.
3. **Verify-and-repair pass (mandatory)**: after encoding, the encoder reconstructs exactly as a decoder would, computes per-element error, and any element whose reconstructed error exceeds the bound is repaired by escalating to a stricter mode for that element (finer quantum → exact/inflation-code correction). Because the repair is deterministic and re-verified to convergence, the bound **cannot** ship violated.

**Rationale**: FR-003/FR-016 make the bound a hard invariant. A verify-and-repair loop is the only way to *guarantee* the bound in the presence of a lossy entropy coder (which must itself be bit-exact), quantization interaction effects (e.g., predictor drift over long horizons), and JPEG AI-style latent transformations. The adaptive-precision escalation satisfies FR-016's "adaptively increasing precision" while the default fine quantization maximizes CR per the user's directive.
- A zero bound is valid (FR-008/FR-017): step ≤ 0 is interpreted as the tightest available representation (may still be lossy if the value cannot be represented exactly, consistent with the clarification).

**Alternatives considered**:
- Trust-the-model (no verification) — rejected: cannot guarantee FR-003.
- Pure fixed-step quantization without repair — rejected: quantized predictor drift can exceed the bound on outliers.
- Reject inputs that can't meet the bound — rejected: contradicts FR-016's adaptive precision approach.

## R5. Transformer model & JPEG AI techniques

**Decision**: v1 uses a compact space-time transformer (attention over spatial patches and time steps) used as a **predictor** (not an end-to-end autoencoder): it predicts values/residuals from neighboring valid data. JPEG AI-inspired elements: learned latent transform, attention/transformer blocks, context modeling for entropy coding. The model is bundled per-package (deterministic, versioned weights, cached/downloaded on demand or shipped small).

**Rationale**: FR-005 mandates a transformer core; the JPEG AI/attention techniques (FR-018) are permitted to maximize CR. A predictor+quantize+verify architecture makes the hard bound directly enforceable — unlike a pure autoencoder whose reconstruction error is only statistically controlled. Bundled/versioned weights keep `get_config` JSON-only while remaining deterministic across encode/decode.

**Alternatives considered**:
- End-to-end learned autoencoder (pure JPEG AI approach) — rejected for v1: error bound only statistical, violating FR-003 by design; revisit later.
- Classical predictors without neural components — rejected: violates FR-005.

## R6. Quantization & entropy coding

**Step size choice**: The quantization step Δ is derived from the bound: Δ ≤ bound (e.g., Δ = bound/2 with verify-and-repair safety margin), so any decoded value lies within bound of the original. Adaptive precision escalation applied per-element regions where the model residual exceeds the quantizer's reach.

**Entropy coder**: v1 ships a **bit-exact** entropy/coding stage: quantized residual symbols coded with a range coder (or bit-packed with per-block modes chosen by the encoder, e.g. raw/blocked-pass-through mode chosen per block by minimizing encoded size — always preserving exactness). Per-block mode selection maximizes CR while remaining exact.

**Rayload structure**: `[bitmask | per-block modes | residual symbols / corrections]`, framed by the container header (see data-model.md, contracts/codec-api.md).

**Rationale**: Exactness of the value path is required for verify-and-repair to be meaningful; the entropy stage must be lossless/`bit-exact` so that verification equals decode. Per-block mode selection (raw vs coded) is the JPEG AI/AI-codec-style rate control knob that maximizes CR.

**Alternatives considered**:
- Lossy entropy coding — rejected: verification would mask it, eroding trust and forcing more repairs, hurting CR.
- Pure fixed global step + raw packing — rejected: leaves CR on the table; adaptive per-block modes maximize CR per the directive.

## R7. Configuration & codec_id

**Decision**: `codec_id = "hoaps-wvpa"`. `get_config()` returns JSON-only params: `{"id": "hoaps-wvpa", "error_bound": <float>, "shape": [t, lat, lon], "missing_value": <float or "nan">, "dtype": "float32"}`. `from_config` restores the codec. Inputs not matching the configured shape are rejected with a clear error. The dtype is fixed to float32 in v1 (HOAPS wvpa is float32; the spec's spirit is physical wvpa values).

**Rationale**: numcodecs stores config separately per the `Codec.get_config` contract; a shape-configured codec is required to interpret a flat buffer as a space-time grid and to size the bitmask. JSON-only keeps `get_config` contract-compliant (R1).

**Alternatives considered**:
- Self-describing stream (shape in container header only) — adopted *additionally* (the container independently records shape/mask/etc. for integrity), but config remains the source of truth for interpretation. This is the numcodecs-idiomatic division.
- float64 support — deferred: HOAPS wvpa is float32; float64 can be added later without contract break via a `dtype` config.

## R8. Testing & verification strategy

**Decision**: Three test tiers mirroring the project structure:
- **Contract tests** (`tests/contract/`): `codec_id`, `get_config`/`from_config` round-trip, JSON-serializability, `encode`/`decode` buffer contract incl. the `out=` parameter, registry integration via `register_codec`/`get_codec`.
- **Unit tests** (`tests/unit/`): mask bitpacking/fidelity, bound validation (reject negative/non-finite, accept zero), container framing/versioning, quantization step derivation, per-element repair escalation.
- **Integration tests** (`tests/integration/`): end-to-end round trips on synthetic HOAPS-like fields (smooth space-time structure + realistic missing masks incl. all-missing, no-missing, land-mask cases), asserting: max error ≤ bound (SC-001), mask bit-identical (SC-002), CR ≥ 2× at reasonable bound (SC-003), monotonic CR w.r.t. bound (SC-004), space-time ≥ spatial-only baseline (SC-007).

**Rationale**: Directly maps to SC-001…SC-007 and the verify-and-repair invariant. Synthetic HOAPS-like fields (smooth geophysical structure, realistic land/ocean missing masks) are used since real HOAPS data is not bundled; test data generation is deterministic.

## R9. Registration and distribution

**Decision**: Ship as a Python package `hoaps-compressor` (src layout, `src/hoaps_compressor/`), expose `HoapsWvpaCodec` in `__init__`, and auto-register with `numcodecs.registry.register_codec` on import. Model weights ship in the package (or are fetched deterministically, versioned).

**Rationale**: Drop-in usability (FR-014): `get_codec({"id": "hoaps-wvpa", ...})` after import. src layout is the modern Python packaging best practice.

**Alternatives considered**:
- Ship without registration (manual import) — rejected: less drop-in.
- Weights fetched at runtime from the internet at decode time — rejected: makes decode depend on network; v1 keeps weights local/versioned (large-weight distribution is a packaging detail tracked in tasks.md).

## R10. Performance approach

**Decision**: Offline, single-machine. Compression may take minutes for 1–50 MB fields (transformer inference + repair); decompression faster than compression. CPU-only support required; GPU optional. Memory bounded by input field size (plus bounded working buffers).

**Rationale**: SC-005 explicitly allows offline processing times; no real-time requirement. Keeping CPU-only support as a floor keeps the library broadly usable; GPU is opportunistic.

**Alternatives considered**:
- Real-time/streaming — out of scope per spec Assumptions.
- Distributed processing — out of scope per spec Assumptions.

## Summary of Resolved Unknowns

| # | Unknown | Resolution |
|---|---------|------------|
| R1 | numcodecs contract members | codec_id, encode, decode(out=), get_config(JSON), from_config; register on import |
| R2 | encode input | contiguous float32 buffer interpreted via config (shape, missing sentinel) |
| R3 | missing values | classify vs sentinel; separate, losslessly stored bitmask; bit-exact restore |
| R4 | bound enforcement | predict → quantize residual → entropy-code → verify-and-repair (mandatory, deterministic) |
| R5 | transformer & JPEG AI | space-time attention transformer as predictor; JPEG AI-style techniques permitted (FR-018); bundled versioned weights |
| R6 | quantization/entropy | step ≤ bound; exact entropy stage; per-block mode selection for max CR |
| R7 | config | codec_id="hoaps-wvpa"; JSON config (error_bound, shape, missing_value, dtype=float32) |
| R8 | testing | contract / unit / integration tiers mapped to SC-001…SC-007 |
| R9 | distribution | Python package, src layout, auto-register, bundled weights |
| R10 | performance | offline, single-machine, CPU floor + optional GPU, minutes per 1–50 MB field |
