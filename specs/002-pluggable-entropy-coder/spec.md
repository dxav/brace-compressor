# Feature Specification: Pluggable Entropy Coder Selection

**Feature Branch**: `002-pluggable-entropy-coder`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "choose zstd (perhaps with dictionary training) or openzl for entropy coding instead of rANS. Choose the one that maximizes the compression ratio. It shall be easy to switch from one entropy coder to another in order to perform compression tests."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare Candidate Entropy Coders (Priority: P1)

A compressor developer wants to compare Zstandard, including an optional trained dictionary, and OpenZL on the same quantized HOAPS WVPA payloads. They run a repeatable benchmark using the same input samples, error bounds, and payload boundaries, then see each candidate's compressed size, compression ratio, and encode/decode time.

**Why this priority**: The feature exists to maximize compression ratio based on evidence rather than an assumed library choice. A fair comparison is the decision-making foundation.

**Independent Test**: Run the benchmark against the representative HOAPS sample and a deterministic synthetic sample at at least three positive error bounds. Verify that every available candidate reports size and timing, and that the benchmark identifies the smallest valid encoded payload.

**Acceptance Scenarios**:

1. **Given** identical quantized residual payloads and benchmark settings, **When** each available candidate is run, **Then** the results report compressed bytes, compression ratio, encode time, decode time, and successful lossless round-trip status.
2. **Given** Zstandard with and without a trained dictionary, **When** both variants are available, **Then** the benchmark reports them as separate candidates rather than combining their results.
3. **Given** a candidate that cannot be used in the current environment, **When** the benchmark runs, **Then** it records the candidate as unavailable with an actionable reason and continues comparing usable candidates.

---

### User Story 2 - Select the Highest-Ratio Coder (Priority: P1)

A compressor maintainer wants the production configuration to use the entropy coder that achieves the best measured compression ratio for the supported HOAPS workload, without weakening the absolute-error or missing-value guarantees.

**Why this priority**: Choosing the best measured coder delivers the storage benefit requested by the feature while retaining the existing scientific correctness contract.

**Independent Test**: Use the benchmark results to select a winner, compress and decompress the same dataset with that winner, and verify that its payload is no larger than every other usable candidate under the defined comparison rules and that all existing error-bound and missing-mask checks still pass.

**Acceptance Scenarios**:

1. **Given** multiple usable candidates, **When** results are evaluated, **Then** the candidate with the smallest total encoded payload for the comparison workload is selected as the recommended production coder; ties are resolved deterministically.
2. **Given** a selected coder, **When** a field is compressed and decompressed, **Then** quantized symbols are recovered exactly and the reconstructed valid values still satisfy the configured absolute error bound.
3. **Given** no candidate is usable, **When** production compression is requested, **Then** the operation fails clearly rather than silently reverting to rANS.

---

### User Story 3 - Switch Entropy Coders for Experiments (Priority: P1)

A researcher wants to select a specific supported entropy coder for a compression run, inspect which coder produced a stream, and decode that stream later without changing the predictor, quantizer, container, or public codec workflow.

**Why this priority**: Easy switching is required to make compression experiments practical and prevents future coder evaluations from requiring invasive pipeline changes.

**Independent Test**: Configure each usable candidate in turn, encode the same field, decode each stream with the corresponding configuration, and verify exact entropy-symbol recovery, correct coder identification, and unchanged scientific guarantees.

**Acceptance Scenarios**:

1. **Given** a valid coder selection, **When** the codec is configured and used, **Then** only that coder is used for the residual payload and the selected coder is recorded in the stream metadata.
2. **Given** a stream created with a supported coder, **When** it is decoded with a matching codec configuration, **Then** decoding succeeds without requiring changes to the predictor or quantizer settings.
3. **Given** an unknown, malformed, or unavailable coder selection, **When** configuration or decoding is attempted, **Then** it fails with a clear error naming the supported choices and does not produce a misleading stream.

## Edge Cases

- A trained Zstandard dictionary may be unavailable, invalid, incompatible with the payload format, or worse than no dictionary. The benchmark must keep the no-dictionary variant and only recommend the dictionary variant when it improves the defined workload result.
- OpenZL may be unavailable on a platform or may not support a required payload. It must be reported as unavailable, not silently substituted.
- Very small, highly irregular, or already incompressible payloads may be larger after entropy coding. The selected coder must preserve the existing raw/passthrough option where it produces the smallest valid payload.
- A coder library or dictionary version may differ between encoding and decoding environments. The stream must identify the coder and required compatibility metadata, and decoding must reject incompatible streams clearly.
- Empty and all-missing inputs produce no valid residual symbols; their existing mask and container behavior must remain unchanged.
- Corrupted or truncated entropy payloads must fail integrity validation and must never yield silently corrupted values.
- Compression benchmarks must use deterministic settings so repeated runs produce comparable sizes and rankings.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support a common lossless entropy-coder contract for encoding and decoding quantized residual symbols without changing predictor, quantizer, error verification, or missing-value behavior.
- **FR-002**: The system MUST provide Zstandard as a supported entropy-coder candidate instead of rANS.
- **FR-003**: The system MUST support evaluating Zstandard with no dictionary and, when a compatible trained dictionary is supplied, with that dictionary as a distinct candidate.
- **FR-004**: The system MUST provide OpenZL as a supported entropy-coder candidate when the runtime environment supplies a compatible OpenZL implementation.
- **FR-005**: The system MUST NOT use rANS as the production or benchmark entropy-coder candidate for this feature.
- **FR-006**: The system MUST allow a caller to select a supported entropy coder through the existing codec configuration without changing the public compression and decompression workflow.
- **FR-007**: The system MUST record the selected entropy coder, variant, and compatibility information in each encoded stream sufficiently for a matching decoder to identify and validate it.
- **FR-008**: The system MUST reject unknown, malformed, unavailable, or incompatible coder selections and streams with clear errors.
- **FR-009**: The system MUST recover quantized residual symbols exactly after entropy decode, including for raw or passthrough payloads.
- **FR-010**: The system MUST preserve the existing hard absolute-error guarantee for every valid value after switching entropy coders; entropy coding MUST NOT add lossy error.
- **FR-011**: The system MUST preserve the existing missing-value mask exactly, including for all-missing and no-valid-value inputs.
- **FR-012**: The system MUST provide a repeatable comparison procedure over identical payloads, datasets, and error bounds for all candidates that are available.
- **FR-013**: The comparison MUST report at least compressed payload size, total encoded-stream size, compression ratio, encode time, decode time, and lossless round-trip status for each candidate.
- **FR-014**: The comparison MUST evaluate Zstandard dictionary training as an optional optimization and MUST retain a no-dictionary baseline.
- **FR-015**: The system MUST select or recommend the candidate with the smallest valid total encoded-stream size for the defined representative workload; ties MUST use a documented deterministic rule.
- **FR-016**: The system MUST retain a valid raw or passthrough representation when it is smaller than a candidate's coded representation, while still identifying the selected entropy-coder path.
- **FR-017**: The system MUST use deterministic coder settings and benchmark inputs so repeated comparisons produce the same candidate ranking within documented measurement tolerance.
- **FR-018**: The system MUST remain compatible with the existing container integrity checks and reject corrupted or truncated entropy payloads.

### Key Entities *(include if feature involves data)*

- **Entropy Coder**: A lossless component that maps quantized residual symbols to bytes and restores the exact symbols during decoding. Supported choices are Zstandard and OpenZL where available.
- **Coder Variant**: A named configuration of an entropy coder, such as Zstandard without a dictionary or Zstandard with a specific trained dictionary and compatibility metadata.
- **Entropy-Coded Payload**: The residual-symbol bytes stored in the existing encoded stream, including coder identification and any required variant metadata.
- **Coder Benchmark Result**: A reproducible record of candidate availability, compressed sizes, compression ratio, encode/decode timings, round-trip status, and failure reason when unavailable.
- **Trained Dictionary**: An optional reusable dictionary produced from representative quantized residual payloads and accepted only when its compatibility and measured benefit are verified.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the representative HOAPS WVPA workload and each declared test error bound, the recommended candidate has a total encoded-stream size no larger than every other usable candidate measured under the same conditions.
- **SC-002**: The benchmark produces complete comparable results for 100% of usable candidates and explicitly records availability or failure reasons for every unusable candidate.
- **SC-003**: Repeating the benchmark with identical inputs and settings produces the same recommended candidate and the same encoded byte size for each deterministic candidate.
- **SC-004**: At least 99.9% of benchmarked non-empty payloads decode to quantized residual symbols byte-for-byte identical to those supplied to the entropy coder; the remaining cases are treated as failures and block recommendation.
- **SC-005**: 100% of valid-value round trips using the recommended coder satisfy the configured absolute error bound, and 100% of missing-value masks remain identical to the source mask.
- **SC-006**: Switching between any two usable supported coders requires changing only the entropy-coder selection or variant configuration; predictor, quantizer, verification, and public codec calls remain unchanged.
- **SC-007**: When a trained Zstandard dictionary is included in an evaluation, its measured result is reported separately and is recommended only when it improves total encoded-stream size over the no-dictionary variant for the representative workload.
- **SC-008**: A coder comparison for a standard 1–50 MB field completes in minutes on a single supported machine, excluding one-time dictionary training, and reports timings for each candidate.

## Assumptions

- The existing HOAPS WVPA codec remains responsible for prediction, quantization, error verification, missing-mask preservation, and container integrity; this feature changes only the lossless residual-byte coding boundary and its configuration.
- Quantized residual symbols are the comparison input, so candidate rankings measure entropy-coder impact rather than differences in prediction or quantization.
- Zstandard is the baseline candidate because it is broadly deployable; OpenZL is optional at runtime and is not assumed to be available on every supported platform.
- Dictionary training is offline and uses representative quantized residual payloads; trained dictionaries are versioned and distributed only when compatibility is verified.
- The recommended coder is selected for the declared representative HOAPS workload, not claimed to be universally optimal for every possible dataset.
- Compression ratio is calculated from the same uncompressed reference and includes coder metadata and payload framing when selecting a winner.
- Backward compatibility with streams produced by the retired rANS implementation is out of scope unless separately specified; new streams use the selected supported coder and fail clearly when an unsupported legacy coder is encountered.
- The existing Python library/API remains the user-facing interface; a separate command-line interface is not required for this feature.
