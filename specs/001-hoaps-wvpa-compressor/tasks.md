---

description: "Task list for HOAPS WVPA Transformer Compressor implementation"
---

# Tasks: HOAPS WVPA Transformer Compressor

**Input**: Design documents from `/specs/001-hoaps-wvpa-compressor/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are included because the design documents (plan.md, quickstart.md) define a mandatory contract/unit/integration test strategy and the hard guarantees (SC-001, SC-002) require automated verification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root (per plan.md structure)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project scaffolding: `pyproject.toml` (name `hoaps-compressor`, Python >=3.11, deps: numcodecs>=0.12, numpy, torch (CPU)), `src/hoaps_compressor/__init__.py`, `tests/` dirs (`tests/contract/`, `tests/unit/`, `tests/integration/`), and `pytest.ini`/`conftest.py`
- [x] T002 Finalize `src/hoaps_compressor/__init__.py` to expose `HoapsWvpaCodec` and auto-register it via `numcodecs.registry.register_codec(HoapsWvpaCodec)` on import (codec class itself arrives with T011; use a temporary stub if needed so import succeeds)
- [x] T003 Add `pyproject.toml` `[project.optional-dependencies] test = ["pytest", "hypothesis"]` and a `[tool.pytest.ini_options]` section

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create `src/hoaps_compressor/bound.py` implementing error-bound validation: `validate_error_bound(value)` MUST raise `ValueError` when `value < 0` or non-finite (per FR-008); `0.0` is valid (tightest allowed bound, FR-017)
- [x] T005 [P] Create `src/hoaps_compressor/mask.py` implementing missing-value mask: `extract_mask(values, missing_value)` returns bool array (`True`=missing) where element equals `missing_value` (or `np.isnan` when `missing_value="nan"`); `pack_mask(mask)` → bitpacked bytes (`ceil(N/8)` bytes); `unpack_mask(packed, n)` → bool array; MUST be bit-exact round-trip (FR-004/FR-015)
- [x] T006 [P] Create `src/hoaps_compressor/container.py` implementing the EncodedStream container framing per contracts/codec-api.md §3: magic `"HWPC"`, container version uint16=1, flags uint16, model version uint32, header-extra length uint32 + JSON header-extra, mask payload length uint64 + payload, residual payload length uint64 + payload, CRC-32 over all preceding bytes; `write_container(...)` and `read_container(buf)` with validation (bad magic/version/checksum → `ValueError`)
- [x] T007 [P] Create `src/hoaps_compressor/quant.py` implementing residual quantization: `derive_step(error_bound)` returns step Δ ≤ `error_bound`; `quantize(residual, step, origin)` → integer symbols; `dequantize(symbols, step, origin)` → floats; quantization error bounded by Δ/2 per element
- [x] T008 [P] Create `src/hoaps_compressor/verify.py` implementing the verify-and-repair loop: `verify_and_repair(orig_valid, decoded_valid, error_bound)` returns `(repaired_symbols, repair_map, max_abs_error)`; MUST escalate precision (finer quantum → exact float32 correction) for any element where `|decoded - orig| > error_bound` and loop until zero violations (FR-003/FR-016); returns `max_abs_error` for FR-012 metrics

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Compress HOAPS WVPA Data with Guaranteed Error Bound (Priority: P1) 🎯 MVP

**Goal**: Encode/decode pipeline that guarantees every reconstructed value is within the absolute error bound (SC-001) and produces a smaller representation (FR-006/SC-003).

**Independent Test**: Encode a synthetic smooth space-time field, decode it, and assert `max(|dec - orig|) <= error_bound` (for bound > 0; bound=0 is exempt per FR-003) and `len(enc) <= 0.5 * orig.nbytes`.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T009 [P] [US1] Contract test for `HoapsWvpaCodec.encode`/`decode` buffer contract (incl. `out=` param) in `tests/contract/test_codec_contract.py`
- [x] T010 [P] [US1] Integration test: bounded round trip on synthetic smooth field asserting `max(|dec-orig|) <= error_bound` and `len(enc) <= 0.5 * orig.nbytes` (SC-003, ≥50% smaller) in `tests/integration/test_error_bound.py`

### Implementation for User Story 1

- [x] T011 [P] [US1] Create `src/hoaps_compressor/codec.py` defining `HoapsWvpaCodec(numcodecs.abc.Codec)` with `codec_id = "hoaps-wvpa"`, constructor `(shape, error_bound, missing_value="nan", dtype="float32")` validating per contracts/codec-api.md §1 (negative/non-finite bound → `ValueError`; `shape` 3 positive ints; `dtype` must be `"float32"`)
- [x] T012 [US1] Implement `HoapsWvpaCodec.encode(buf)` in `src/hoaps_compressor/codec.py`: validate buffer size == `4*prod(shape)` and C-contiguous (else `ValueError`); extract mask (T005); predict valid values (placeholder predictor, transformer arrives in US4); quantize residuals (T007) — when `error_bound == 0`, skip verify-and-repair (exempt per FR-003) and use tightest available representation; entropy-code (placeholder in US4); run verify-and-repair (T008) when bound > 0; frame container (T006); return bytes
- [x] T013 [US1] Implement `HoapsWvpaCodec.decode(buf, out=None)` in `src/hoaps_compressor/codec.py`: validate header/checksum (T006); restore mask bit-exactly; entropy-decode + dequantize residuals; write sentinel into missing positions; honor `out=` (must be exactly `4*prod(shape)` bytes else `ValueError`); return float32 array of configured `shape`
- [x] T014 [US1] Add FR-012 metrics to encode in `src/hoaps_compressor/codec.py`: record `max_abs_error` (verified ≤ bound), `n_repaired`, `uncompressed_size`, `compressed_size`, `cr` in header-extra JSON per contracts/codec-api.md §5

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Preserve Missing Values (Priority: P1)

**Goal**: Missing-value mask preserved bit-exactly through encode/decode (SC-002); handles all-missing and no-valid-value inputs (FR-009/FR-010).

**Independent Test**: Encode a field with a known missing mask (incl. all-missing), decode, and assert the mask is identical and no valid↔missing conversion occurred.

### Tests for User Story 2 ⚠️

- [x] T015 [P] [US2] Unit test for mask bitpacking fidelity (round-trip, all-missing, no-missing, land-strip masks) in `tests/unit/test_mask.py`
- [x] T016 [P] [US2] Integration test: mask identity + no valid↔missing conversion on fields with missing values in `tests/integration/test_missing_values.py`

### Implementation for User Story 2

- [x] T017 [P] [US2] Wire mask payload into container encode/decode in `src/hoaps_compressor/container.py`: store bitpacked mask losslessly; when all elements missing, write empty residual payload (no value payload) per data-model.md `MissingMask` state transitions
- [x] T018 [US2] Handle all-missing and no-valid-value inputs in `src/hoaps_compressor/codec.py`: all-missing → mask all-True, empty residual payload, valid decompressible output (FR-009/FR-010); stray non-finite values not matching sentinel treated as missing and counted (FR-011)
- [x] T019 [US2] Ensure decode restores sentinel exactly into missing positions in `src/hoaps_compressor/codec.py` (no valid value converted to missing and vice versa, FR-004)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Configurable Error Bound (Priority: P2)

**Goal**: Users configure different error bounds; each honored (SC-001), looser bound yields smaller size (SC-004), invalid bounds rejected (SC-006).

**Independent Test**: Encode same field at two bounds; assert both respect their bound and `len(enc_loose) <= len(enc_tight)`; assert negative/non-finite bounds raise `ValueError`.

### Tests for User Story 3 ⚠️

- [x] T020 [P] [US3] Contract test for `get_config`/`from_config` round-trip and JSON-serializability (incl. `"id"` field) in `tests/contract/test_config.py`
- [x] T021 [P] [US3] Integration test: bound sweep (e.g. 0.01/0.05/0.2) asserting bound respected (bound > 0) and CR monotonicity, with the reference bound yielding `len(enc) <= 0.5 * orig.nbytes` (SC-003) in `tests/integration/test_configurable_bound.py`

### Implementation for User Story 3

- [x] T022 [P] [US3] Implement `get_config()` and `from_config(config)` in `src/hoaps_compressor/codec.py` per contracts/codec-api.md §2: JSON-serializable dict with `"id": "hoaps-wvpa"`, `shape`, `error_bound`, `missing_value`, `dtype`; `from_config` validates `"id"` and reconstructs behaviorally-identical codec
- [x] T023 [US3] Ensure quantization step derives from the configured bound in `src/hoaps_compressor/quant.py` (`derive_step(error_bound)` with step ≤ bound) so each bound is honored and looser bounds yield smaller residuals (SC-004)
- [x] T024 [US3] Add bound validation coverage in `src/hoaps_compressor/bound.py` for `error_bound=0.0` accepted as tightest allowed bound (FR-017/SC-006)

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Transformer-Based Compression (Priority: P2)

**Goal**: Space-time transformer predictor + bit-exact entropy coding (FR-005, FR-018); space-time CR ≥ spatial-only baseline (SC-007).

**Independent Test**: Confirm pipeline uses transformer predictor and entropy coding; assert space-time codec CR ≥ spatial-only codec CR at equal bound on synthetic fields.

### Tests for User Story 4 ⚠️

- [x] T025 [P] [US4] Unit test for transformer predictor determinism (same weights → same output) in `tests/unit/test_transformer.py`
- [x] T026 [P] [US4] Integration test: space-time CR ≥ spatial-only baseline at equal bound in `tests/integration/test_space_time_cr.py`

### Implementation for User Story 4

- [x] T027 [P] [US4] Create `src/hoaps_compressor/model/transformer.py` implementing a space-time transformer predictor (attention over spatial patches across time steps, JPEG AI-inspired transform/attention blocks) predicting values/residuals from neighboring valid data; deterministic for bundled versioned weights (FR-005, FR-018)
- [x] T028 [P] [US4] Create `src/hoaps_compressor/model/entropy.py` implementing a bit-exact entropy coder: per-block mode selection (`RAW`/`RANGE_CODED`/`PASSTHROUGH`) chosen at encode to minimize encoded size; MUST be lossless so decode reproduces quantized symbols exactly (research.md R6)
- [x] T029 [US4] Integrate transformer predictor (T027) and entropy coder (T028) into `src/hoaps_compressor/codec.py` encode/decode paths, replacing placeholders from US1; record model version in container header (T006) and validate on decode (mismatch → `ValueError`)
- [x] T030 [US4] Add optional outer lossless compression of payloads (container `flags` bit0) in `src/hoaps_compressor/container.py` to maximize CR without touching guarantees (FR-018)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T031 [P] Add unit tests for `bound.py` (negative/non-finite/zero) and `quant.py` (step derivation, quantization error ≤ Δ/2) in `tests/unit/test_bound.py`, `tests/unit/test_quant.py`
- [x] T032 [P] Add container framing/versioning/checksum unit tests in `tests/unit/test_container.py`
- [x] T033 Run `quickstart.md` validation scenarios end-to-end (Scenarios 1–5) and confirm all assertions pass
- [x] T034 Code cleanup and refactoring across `src/hoaps_compressor/` (consistent error messages, docstrings)
- [x] T035 Performance sanity check: encode/decode of a ~1–50 MB synthetic field completes in minutes on CPU (SC-005); document measured times in `docs/performance.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete
- **Enhancement (Phase 8)**: Depends on all user stories + Polish complete; strictly sequential T036→T046 (each task depends on the previous)

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Uses mask (T005) and container (T006) from Foundational; independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Uses bound (T004) and quant (T007); independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Replaces US1 placeholders (T012/T013) but is independently testable via its own predictor/entropy units
- **Enhancement (Phase 8)**: Depends on US4 (transformer predictor) and Polish; sequential chain T036→T046

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Contract test for HoapsWvpaCodec.encode/decode in tests/contract/test_codec_contract.py"
Task: "Integration test for bounded round trip in tests/integration/test_error_bound.py"

# Launch implementation tasks:
Task: "Create HoapsWvpaCodec class in src/hoaps_compressor/codec.py"
Task: "Implement encode in src/hoaps_compressor/codec.py"
```

---

## Phase 8: Intensive Transformer & Attention Prediction (Enhancement)

**Branch**: `001-hoaps-wvpa-attention` | **Date**: 2026-09-15

**Purpose**: Make the transformer/attention the primary predictor during the causal scan (not just a cold-start prior) to raise compression ratio (SC-008), while preserving the hard error bound, mask fidelity, and encode/decode determinism. See plan.md "Phase 8 (Enhancement)" and research.md R11.

**Dependency chain**: T036 → T037 → T038 → T039 → T040 → T041 → T042 → T043 → T044 → T045. Tasks without `[P]` depend on the preceding task in this chain.

- [x] T036 Fix `out_scale` initialization in `src/hoaps_compressor/model/transformer.py`: currently `_init_weights()` zeroes the scalar `out_scale`, so the prior is near-constant (`tanh(0)=0`). Initialize it to a non-zero value (e.g. 1.0) so the transformer output is not collapsed; add a unit test in `tests/unit/test_transformer.py` asserting the prior is not constant across a varied mask
- [x] T037 Add a HOAPS training utility `scripts/train_prior.py` implementing self-supervised masked-cell prediction: sample a mask, hide valid values, predict them from the 6-channel context, loss = MSE (optionally Huber); train on `data/wvpa_2020-08-01_07.npy` (and any additional HOAPS slices); export weights via `TransformerPredictor.serialize_weights()` to a versioned blob. Note: `data/wvpa_2020-08-01_07.npy` is a real-data file not bundled in a fresh checkout — document the data prerequisite and provide a synthetic-data fallback so the script is runnable without it. **Outcome**: the trained prior gives identical CR to the random init (see T039/T044); the utility remains available for explicit use but is not the default
- [x] T038 Add trained-weight loading path: `TransformerPredictor.load_weights(blob)` already exists; add a package-level default weights file (e.g. `src/hoaps_compressor/model/weights/prior_v3.hwpm`) and load it in `TransformerPredictor.__init__` when present; bump `MODEL_VERSION` to 3; keep deterministic random init as fallback when weights are absent. **Outcome**: the trained weights file was removed as the default because it provides no CR benefit (identical CR to random init); `load_weights()` remains available for explicit use
- [x] T039 Benchmark the trained base prior alone: run `scripts/compress_stats.py --input data/wvpa_2020-08-01_07.npy --bound <b>` before/after training and record CR, max error, encode/decode time in `docs/performance.md`. **Outcome**: at the same quantization step, random and trained priors give identical CR (16.34× at bound 0.05, 11.07× at bound 0.01); the trained prior provides no CR benefit
- [x] T040 Add block-local causal attention predictor in Python: in `src/hoaps_compressor/model/transformer.py`, add a method that processes valid cells in causal blocks of a fixed default size (256 positions), feeding reconstructed temporal/spatial neighbors + mask + positional encodings with a causal/local attention mask; output becomes the per-cell prediction; keep the existing weighted-average predictor as fallback
- [x] T041 Add unit tests for the block-local causal predictor in `tests/unit/test_transformer.py`: determinism (same state → same prediction), causality (a cell never attends to a not-yet-reconstructed cell), and equivalence with the weighted-average fallback on cold starts
- [x] T042 Integrate the block-local causal predictor into `src/hoaps_compressor/codec.py` encode/decode causal scans (`_causal_scan_encode`/`_causal_scan_decode`), replacing the fixed weighted average where the transformer is available; ensure encode and decode walk the same blocks in the same order (determinism)
- [x] T043 Port the finalized block-local causal predictor to Rust (`rust/hoaps_scan/src/lib.rs`) for speed, keeping it bit-exact with the Python reference; dispatch to the accelerated path when available (TorchScript is a fallback if the Rust port proves impractical). **Deferred**: the block predictor is experimental and currently lowers CR (untrained `block_in_proj`); the Rust port is deferred until a trained block predictor demonstrates a CR win (see docs/performance.md)
- [x] T044 Benchmark CR and runtime on `data/wvpa_2020-08-01_07.npy` at multiple bounds; record results in `docs/performance.md`; confirm CR strictly higher than the pre-enhancement baseline at the same bound (SC-008). **Outcome**: SC-008 is NOT met by the trained prior — the CR increase from 11.67× to 16.34× was due to the quantization-step change (`Δ = 2·bound`), not the prior. The trained prior gives identical CR to the random init at the same step. The quantization change is the real CR driver
- [x] T045 Update `docs/architecture.md` and `specs/001-hoaps-wvpa-compressor/data-model.md` §5 to document the new block-local causal predictor (replacing the "fixed weighted average" description) and the trained-weight loading path
- [x] T046 Re-run the full test suite (`pytest tests -q`) and confirm all existing tests still pass (hard error bound, mask fidelity, determinism, config, container) with the new predictor active
- [x] T047 Rebalance the causal-scan neighbor weights to a spatial-weighted stencil in `src/hoaps_compressor/codec.py` and `rust/hoaps_scan/src/lib.rs` (bit-exact): from `(temporal 4, left 2, top 2, diag 1, diag 1)` to `(left 5, top 5, top-left 2, top-right 2, temporal 1)`, because real HOAPS wvpa has much stronger spatial than temporal correlation (left-neighbor MAE 0.80 vs temporal MAE 2.05). **Outcome**: CR rises from 16.34× to 17.03× at bound 0.05 on `data/wvpa_2020-08-01_07.npy` (residual entropy 5.70 → 5.44 bits/symbol); Rust and Python paths verified byte-identical; all 74 tests pass

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (guaranteed error bound + smaller output)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 (missing-value preservation) → Test independently → Deploy/Demo
4. Add User Story 3 (configurable bound) → Test independently → Deploy/Demo
5. Add User Story 4 (transformer + entropy) → Test independently → Deploy/Demo
6. Add Enhancement (Phase 8: intensive transformer/attention prediction) → Test independently → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
