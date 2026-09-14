# Feature Specification: HOAPS WVPA Transformer Compressor

**Feature Branch**: `001-hoaps-wvpa-compressor`

**Created**: 2026-09-14

**Status**: Draft

**Input**: User description: "a data compression software dedicated to the compression of Earth climate data and specifically HOAPS https://climatedataguide.ucar.edu/climate-data/hoaps-hamburg-ocean-atmosphere-parameters-and-fluxes-satellite-data, water vapor wvpa variable. The compression shall use a transformer. It shall have an absolute error bound. The compressor must preserve the missing values and must not violate the error bound."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compress HOAPS WVPA Data with Guaranteed Error Bound (Priority: P1)

A climate data scientist has a HOAPS dataset containing the water vapor (wvpa) variable and wants to reduce its storage footprint. They run the compressor on the wvpa data, providing an absolute error bound. The compressor produces a compressed representation that, when decompressed, reconstructs the water vapor values such that no reconstructed value deviates from the original by more than the specified absolute error bound.

**Why this priority**: This is the core value of the feature — lossy compression with a hard, verifiable accuracy guarantee. Without it, the feature has no purpose.

**Independent Test**: Can be fully tested by compressing a HOAPS wvpa sample, decompressing it, and verifying that the maximum absolute difference between original and reconstructed values is at or below the specified error bound. Delivers the primary value of storage reduction with guaranteed accuracy.

**Acceptance Scenarios**:

1. **Given** a HOAPS wvpa dataset and a specified absolute error bound, **When** the data is compressed and then decompressed, **Then** every reconstructed value is within the error bound of its original value.
2. **Given** a HOAPS wvpa dataset, **When** the data is compressed, **Then** the compressed representation is smaller than the original uncompressed data.

---

### User Story 2 - Preserve Missing Values (Priority: P1)

The HOAPS wvpa variable contains missing values (e.g., over land or where no satellite retrieval is possible). A climate data scientist compresses the data and expects that the locations and values of missing data are preserved exactly — missing values must not be altered, filled, or turned into valid numbers, and valid values must not be turned into missing values.

**Why this priority**: Missing-value fidelity is a correctness requirement for climate data. Corrupting missing-value masks would invalidate downstream scientific analysis, so it is equally critical as the error bound.

**Independent Test**: Can be fully tested by compressing a wvpa sample containing known missing values, decompressing it, and verifying that the missing-value mask is identical to the original and that no valid value was converted to missing (or vice versa).

**Acceptance Scenarios**:

1. **Given** a HOAPS wvpa dataset containing missing values, **When** the data is compressed and decompressed, **Then** the set of locations flagged as missing is identical to the original.
2. **Given** a HOAPS wvpa dataset, **When** the data is compressed and decompressed, **Then** no valid data value is converted into a missing value and no missing value is converted into a valid data value.

---

### User Story 3 - Configurable Error Bound (Priority: P2)

A climate data scientist wants to trade off compression ratio against accuracy. They provide different absolute error bounds for different use cases (e.g., a tight bound for research-grade analysis, a looser bound for archival storage) and expect the compressor to honor each bound.

**Why this priority**: Configurability makes the tool broadly useful across accuracy/storage trade-offs, but it builds on the core compression capability.

**Independent Test**: Can be fully tested by compressing the same dataset with two different error bounds and verifying that (a) each result respects its own bound and (b) the looser bound yields a smaller compressed size than the tighter bound.

**Acceptance Scenarios**:

1. **Given** a HOAPS wvpa dataset, **When** it is compressed with a tighter error bound and separately with a looser error bound, **Then** both results respect their respective bounds.
2. **Given** a HOAPS wvpa dataset, **When** it is compressed with a looser error bound, **Then** the resulting compressed size is no larger than when compressed with a tighter error bound.

---

### User Story 4 - Transformer-Based Compression (Priority: P2)

The compression algorithm is based on a transformer model. The compressor uses a transformer to model and encode the water vapor field, achieving compression while respecting the error bound. This is an internal design requirement that enables high compression ratios on smooth geophysical fields.

**Why this priority**: The transformer is the mandated technical approach, but its value is realized only through the error-bound and missing-value guarantees above.

**Independent Test**: Can be tested by confirming the compression pipeline is transformer-based and that it still satisfies the error-bound and missing-value preservation guarantees on a wvpa sample.

**Acceptance Scenarios**:

1. **Given** a HOAPS wvpa dataset, **When** it is compressed, **Then** the compression pipeline uses a transformer-based model as its core encoding mechanism.
2. **Given** a transformer-based compression pipeline, **When** it is run on a wvpa sample, **Then** it still satisfies the absolute error bound and preserves missing values.

---

### Edge Cases

- What happens when the entire dataset (or a large region) consists of missing values? The compressor must handle all-missing inputs without error and preserve the mask exactly.
- How does the system handle an error bound of zero? It must either produce lossless output or reject the value with a clear message, since a zero bound implies exact reconstruction.
- How does the system handle a negative or non-finite error bound? It must reject invalid bounds with a clear error.
- How does the system handle data containing extreme or outlier values? Reconstructed values must still respect the error bound.
- How does the system handle empty input or a dataset with no valid (non-missing) values? It must not crash and must produce a valid, decompressible output.
- How does the system handle NaN or other non-finite values that are not the designated missing-value sentinel? Behavior must be defined (e.g., treated as missing or rejected).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept a HOAPS wvpa dataset as input and produce a compressed representation that can be decompressed back into a wvpa dataset.
- **FR-002**: The system MUST accept a user-specified absolute error bound that defines the maximum allowed absolute difference between each original and reconstructed value.
- **FR-003**: The system MUST guarantee that, after decompression, every reconstructed value is within the specified absolute error bound of its original value.
- **FR-004**: The system MUST preserve missing values exactly: the missing-value mask after decompression MUST be identical to the original, and no value may change between valid and missing status.
- **FR-005**: The system MUST use a transformer-based model as the core of its compression algorithm.
- **FR-006**: The system MUST produce a compressed representation that is smaller than the original uncompressed data for typical HOAPS wvpa inputs.
- **FR-007**: The system MUST support configurable error bounds so users can trade off accuracy against compression ratio.
- **FR-008**: The system MUST reject invalid error bounds (negative, zero, or non-finite) with a clear error message.
- **FR-009**: The system MUST handle inputs that are entirely or largely missing values without failing, and MUST preserve the missing-value mask exactly in such cases.
- **FR-010**: The system MUST handle empty inputs and inputs with no valid values gracefully, producing a valid, decompressible output.
- **FR-011**: The system MUST define and document the behavior for non-finite values (e.g., NaN) that are not the designated missing-value sentinel.
- **FR-012**: The system MUST report the achieved compression ratio and confirm that the error bound was respected for each compression run.

### Key Entities *(include if feature involves data)*

- **HOAPS WVPA Dataset**: The input/output climate data. Represents satellite-derived water vapor (wvpa) over the ocean, typically as a gridded field with a spatial and temporal structure. Key attributes: grid dimensions, per-cell water vapor values, and a missing-value mask.
- **Missing-Value Mask**: The set of grid locations flagged as missing (no valid retrieval). Must be preserved exactly through compression and decompression.
- **Absolute Error Bound**: The user-supplied maximum allowed absolute difference between original and reconstructed values. A scalar configuration parameter.
- **Compressed Representation**: The output artifact produced by the compressor, smaller than the original, from which the original (within the error bound) can be reconstructed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For any HOAPS wvpa input and any valid error bound, 100% of reconstructed values are within the specified absolute error bound of their original values.
- **SC-002**: The missing-value mask is preserved with 100% accuracy: after decompression, the mask is identical to the original and no value changes between valid and missing status.
- **SC-003**: For typical HOAPS wvpa inputs, the compressed representation is at least 50% smaller than the original uncompressed data at a reasonable error bound.
- **SC-004**: Users can configure the error bound and observe that a looser bound yields a smaller compressed size than a tighter bound on the same data.
- **SC-005**: The compressor completes compression and decompression of a standard HOAPS wvpa field within a time acceptable for offline scientific processing (no real-time requirement).
- **SC-006**: Invalid error bounds (negative, zero, non-finite) are rejected with a clear error message 100% of the time.

## Assumptions

- The error bound is a user-supplied configuration parameter; a reasonable default (e.g., a small fraction of the data's value range) is provided when the user does not specify one.
- The input and output data use a standard climate data format (e.g., NetCDF) with a well-defined missing-value sentinel, consistent with how HOAPS data is distributed.
- Compression is lossy (values are approximated within the error bound); the error bound is the mechanism that controls accuracy.
- The transformer model is trained or configured to operate on the wvpa field; model training data and the specific transformer architecture are implementation details outside the scope of this specification.
- The primary target is offline scientific processing of HOAPS wvpa data; real-time or streaming compression is out of scope for v1.
- The compressor is expected to run on a single machine with reasonable compute resources; distributed processing is out of scope for v1.
- The error bound is interpreted as an absolute (not relative) error on the physical water vapor values.
