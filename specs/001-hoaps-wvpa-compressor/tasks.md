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

- [ ] T001 Create project scaffolding: `pyproject.toml` (name `hoaps-compressor`, Python >=3.11, deps: numcodecs>=0.12, numpy, torch (CPU)), `src/hoaps_compressor/__init__.py`, `tests/` dirs (`tests/contract/`, `tests/unit/`, `tests/integration/`), and `pytest.ini`/`conftest.py`
- [ ] T002 Finalize `src/hoaps_compressor/__init__.py` to expose `HoapsWvpaCodec` and auto-register it via `numcodecs.registry.register_codec(HoapsWvpaCodec)` on import (codec class itself arrives with T011; use a temporary stub if needed so import succeeds)
- [ ] T003 Add `pyproject.toml` `[project.optional-dependencies] test = ["pytest", "hypothesis"]` and a `[tool.pytest.ini_options]` section

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create `src/hoaps_compressor/bound.py` implementing error-bound validation: `validate_error_bound(value)` MUST raise `ValueError` when `value < 0` or non-finite (per FR-008); `0.0` is valid (tightest allowed bound, FR-017)
- [ ] T005 [P] Create `src/hoaps_compressor/mask.py` implementing missing-value mask: `extract_mask(values, missing_value)` returns bool array (`True`=missing) where element equals `missing_value` (or `np.isnan` when `missing_value="nan"`); `pack_mask(mask)` → bitpacked bytes (`ceil(N/8)` bytes); `unpack_mask(packed, n)` → bool array; MUST be bit-exact round-trip (FR-004/FR-015)
- [ ] T006 [P] Create `src/hoaps_compressor/container.py` implementing the EncodedStream container framing per contracts/codec-api.md §3: magic `"HWPC"`, container version uint16=1, flags uint16, model version uint32, header-extra length uint32 + JSON header-extra, mask payload length uint64 + payload, residual payload length uint64 + payload, CRC-32 over all preceding bytes; `write_container(...)` and `read_container(buf)` with validation (bad magic/version/checksum → `ValueError`)
- [ ] T007 [P] Create `src/hoaps_compressor/quant.py` implementing residual quantization: `derive_step(error_bound)` returns step Δ ≤ `error_bound`; `quantize(residual, step, origin)` → integer symbols; `dequantize(symbols, step, origin)` → floats; quantization error bounded by Δ/2 per element
- [ ] T008 [P] Create `src/hoaps_compressor/verify.py` implementing the verify-and-repair loop: `verify_and_repair(orig_valid, decoded_valid, error_bound)` returns `(repaired_symbols, repair_map, max_abs_error)`; MUST escalate precision (finer quantum → exact float32 correction) for any element where `|decoded - orig| > error_bound` and loop until zero violations (FR-003/FR-016); returns `max_abs_error` for FR-012 metrics

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Compress HOAPS WVPA Data with Guaranteed Error Bound (Priority: P1) 🎯 MVP

**Goal**: Encode/decode pipeline that guarantees every reconstructed value is within the absolute error bound (SC-001) and produces a smaller representation (FR-006/SC-003).

**Independent Test**: Encode a synthetic smooth space-time field, decode it, and assert `max(|dec - orig|) <= error_bound` (for bound > 0; bound=0 is exempt per FR-003) and `len(enc) <= 0.5 * orig.nbytes`.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T009 [P] [US1] Contract test for `HoapsWvpaCodec.encode`/`decode` buffer contract (incl. `out=` param) in `tests/contract/test_codec_contract.py`
- [ ] T010 [P] [US1] Integration test: bounded round trip on synthetic smooth field asserting `max(|dec-orig|) <= error_bound` and `len(enc) <= 0.5 * orig.nbytes` (SC-003, ≥50% smaller) in `tests/integration/test_error_bound.py`

### Implementation for User Story 1

- [ ] T011 [P] [US1] Create `src/hoaps_compressor/codec.py` defining `HoapsWvpaCodec(numcodecs.abc.Codec)` with `codec_id = "hoaps-wvpa"`, constructor `(shape, error_bound, missing_value="nan", dtype="float32")` validating per contracts/codec-api.md §1 (negative/non-finite bound → `ValueError`; `shape` 3 positive ints; `dtype` must be `"float32"`)
- [ ] T012 [US1] Implement `HoapsWvpaCodec.encode(buf)` in `src/hoaps_compressor/codec.py`: validate buffer size == `4*prod(shape)` and C-contiguous (else `ValueError`); extract mask (T005); predict valid values (placeholder predictor, transformer arrives in US4); quantize residuals (T007) — when `error_bound == 0`, skip verify-and-repair (exempt per FR-003) and use tightest available representation; entropy-code (placeholder in US4); run verify-and-repair (T008) when bound > 0; frame container (T006); return bytes
- [ ] T013 [US1] Implement `HoapsWvpaCodec.decode(buf, out=None)` in `src/hoaps_compressor/codec.py`: validate header/checksum (T006); restore mask bit-exactly; entropy-decode + dequantize residuals; write sentinel into missing positions; honor `out=` (must be exactly `4*prod(shape)` bytes else `ValueError`); return float32 array of configured `shape`
- [ ] T014 [US1] Add FR-012 metrics to encode in `src/hoaps_compressor/codec.py`: record `max_abs_error` (verified ≤ bound), `n_repaired`, `uncompressed_size`, `compressed_size`, `cr` in header-extra JSON per contracts/codec-api.md §5

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Preserve Missing Values (Priority: P1)

**Goal**: Missing-value mask preserved bit-exactly through encode/decode (SC-002); handles all-missing and no-valid-value inputs (FR-009/FR-010).

**Independent Test**: Encode a field with a known missing mask (incl. all-missing), decode, and assert the mask is identical and no valid↔missing conversion occurred.

### Tests for User Story 2 ⚠️

- [ ] T015 [P] [US2] Unit test for mask bitpacking fidelity (round-trip, all-missing, no-missing, land-strip masks) in `tests/unit/test_mask.py`
- [ ] T016 [P] [US2] Integration test: mask identity + no valid↔missing conversion on fields with missing values in `tests/integration/test_missing_values.py`

### Implementation for User Story 2

- [ ] T017 [P] [US2] Wire mask payload into container encode/decode in `src/hoaps_compressor/container.py`: store bitpacked mask losslessly; when all elements missing, write empty residual payload (no value payload) per data-model.md `MissingMask` state transitions
- [ ] T018 [US2] Handle all-missing and no-valid-value inputs in `src/hoaps_compressor/codec.py`: all-missing → mask all-True, empty residual payload, valid decompressible output (FR-009/FR-010); stray non-finite values not matching sentinel treated as missing and counted (FR-011)
- [ ] T019 [US2] Ensure decode restores sentinel exactly into missing positions in `src/hoaps_compressor/codec.py` (no valid value converted to missing and vice versa, FR-004)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Configurable Error Bound (Priority: P2)

**Goal**: Users configure different error bounds; each honored (SC-001), looser bound yields smaller size (SC-004), invalid bounds rejected (SC-006).

**Independent Test**: Encode same field at two bounds; assert both respect their bound and `len(enc_loose) <= len(enc_tight)`; assert negative/non-finite bounds raise `ValueError`.

### Tests for User Story 3 ⚠️

- [ ] T020 [P] [US3] Contract test for `get_config`/`from_config` round-trip and JSON-serializability (incl. `"id"` field) in `tests/contract/test_config.py`
- [ ] T021 [P] [US3] Integration test: bound sweep (e.g. 0.01/0.05/0.2) asserting bound respected (bound > 0) and CR monotonicity, with the reference bound yielding `len(enc) <= 0.5 * orig.nbytes` (SC-003) in `tests/integration/test_configurable_bound.py`

### Implementation for User Story 3

- [ ] T022 [P] [US3] Implement `get_config()` and `from_config(config)` in `src/hoaps_compressor/codec.py` per contracts/codec-api.md §2: JSON-serializable dict with `"id": "hoaps-wvpa"`, `shape`, `error_bound`, `missing_value`, `dtype`; `from_config` validates `"id"` and reconstructs behaviorally-identical codec
- [ ] T023 [US3] Ensure quantization step derives from the configured bound in `src/hoaps_compressor/quant.py` (`derive_step(error_bound)` with step ≤ bound) so each bound is honored and looser bounds yield smaller residuals (SC-004)
- [ ] T024 [US3] Add bound validation coverage in `src/hoaps_compressor/bound.py` for `error_bound=0.0` accepted as tightest allowed bound (FR-017/SC-006)

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Transformer-Based Compression (Priority: P2)

**Goal**: Space-time transformer predictor + bit-exact entropy coding (FR-005, FR-018); space-time CR ≥ spatial-only baseline (SC-007).

**Independent Test**: Confirm pipeline uses transformer predictor and entropy coding; assert space-time codec CR ≥ spatial-only codec CR at equal bound on synthetic fields.

### Tests for User Story 4 ⚠️

- [ ] T025 [P] [US4] Unit test for transformer predictor determinism (same weights → same output) in `tests/unit/test_transformer.py`
- [ ] T026 [P] [US4] Integration test: space-time CR ≥ spatial-only baseline at equal bound in `tests/integration/test_space_time_cr.py`

### Implementation for User Story 4

- [ ] T027 [P] [US4] Create `src/hoaps_compressor/model/transformer.py` implementing a space-time transformer predictor (attention over spatial patches across time steps, JPEG AI-inspired transform/attention blocks) predicting values/residuals from neighboring valid data; deterministic for bundled versioned weights (FR-005, FR-018)
- [ ] T028 [P] [US4] Create `src/hoaps_compressor/model/entropy.py` implementing a bit-exact entropy coder: per-block mode selection (`RAW`/`RANGE_CODED`/`PASSTHROUGH`) chosen at encode to minimize encoded size; MUST be lossless so decode reproduces quantized symbols exactly (research.md R6)
- [ ] T029 [US4] Integrate transformer predictor (T027) and entropy coder (T028) into `src/hoaps_compressor/codec.py` encode/decode paths, replacing placeholders from US1; record model version in container header (T006) and validate on decode (mismatch → `ValueError`)
- [ ] T030 [US4] Add optional outer lossless compression of payloads (container `flags` bit0) in `src/hoaps_compressor/container.py` to maximize CR without touching guarantees (FR-018)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T031 [P] Add unit tests for `bound.py` (negative/non-finite/zero) and `quant.py` (step derivation, quantization error ≤ Δ/2) in `tests/unit/test_bound.py`, `tests/unit/test_quant.py`
- [ ] T032 [P] Add container framing/versioning/checksum unit tests in `tests/unit/test_container.py`
- [ ] T033 Run `quickstart.md` validation scenarios end-to-end (Scenarios 1–5) and confirm all assertions pass
- [ ] T034 Code cleanup and refactoring across `src/hoaps_compressor/` (consistent error messages, docstrings)
- [ ] T035 Performance sanity check: encode/decode of a ~1–50 MB synthetic field completes in minutes on CPU (SC-005); document measured times in `docs/performance.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Uses mask (T005) and container (T006) from Foundational; independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Uses bound (T004) and quant (T007); independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Replaces US1 placeholders (T012/T013) but is independently testable via its own predictor/entropy units

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
6. Each story adds value without breaking previous stories

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
