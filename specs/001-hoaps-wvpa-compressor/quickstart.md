# Quickstart: HOAPS WVPA Transformer Compressor

**Branch**: `001-hoaps-wvpa-attention` | **Date**: 2026-09-15

Runnable validation scenarios proving the feature end-to-end. Implementation details live in [../plan.md](../plan.md) and `tasks.md`; entity/byte-level details in [data-model.md](data-model.md) and [contracts/codec-api.md](contracts/codec-api.md).

## Prerequisites

- Python 3.11+
- Working install of the package in development mode and runtime deps:

```bash
pip install -e ".[test]"       # installs numcodecs, numpy, torch (CPU), pytest
```

- No network or GPU required; no real HOAPS data files needed (tests generate deterministic HOAPS-like synthetic fields).

## Scenario 1 — Codec contract smoke test (numcodecs drop-in)

Proves the codec implements the `numcodecs.Codec` contract (FR-014): registry lookup, config round-trip, JSON-serializable config.

```python
import numpy as np
import hoaps_compressor  # registers codec on import
import numcodecs.registry

codec = numcodecs.registry.get_codec(
    {"id": "hoaps-wvpa", "shape": (4, 8, 16), "error_bound": 0.05}
)
cfg = codec.get_config()
assert cfg["id"] == "hoaps-wvpa"
import json; json.dumps(cfg)  # must be JSON-serializable
codec2 = numcodecs.registry.get_codec(cfg)
```

**Expected**: no assertion errors; `codec2` behaviorally identical to `codec`.

Run via the bundled contract suite:

```bash
pytest tests/contract -v
```

**Expected**: all contract tests pass (codec_id, get_config/from_config round-trip, encode/decode buffers incl. `out=`, registry integration).

## Scenario 2 — Error-bounded round trip with missing values

Proves SC-001 (bound respected) and SC-002 (mask bit-exact).

```python
import numpy as np
import hoaps_compressor  # or: from hoaps_compressor import HoapsWvpaCodec

rng = np.random.default_rng(42)
shape = (8, 32, 64)
bound = 0.05
# Smooth space-time field + realistic land/edge missing mask
field = ...  # e.g. synthetic smooth wvpa-like data, float32, ~250 K/kg units
mask = np.zeros(shape, dtype=bool)
mask[:, :4, :] = True          # "land" strip missing
orig = np.where(mask, np.nan, field).astype("float32")

codec = hoaps_compressor.HoapsWvpaCodec(shape=shape, error_bound=0.05, missing_value="nan")
enc = codec.encode(orig)
dec = codec.decode(enc)

valid = ~np.isnan(orig)
assert np.array_equal(np.isnan(dec), np.isnan(orig))            # mask identical (SC-002)
assert np.all(np.abs((dec - orig)[valid]) <= 0.05 + 1e-6)       # bound respected (SC-001)
import sys; assert len(enc) < orig.nbytes                        # smaller (FR-006/SC-003)
print("CR:", orig.nbytes / len(enc))
```

**Expected**: `CR` comfortably ≥ 2 on smooth synthetic fields; mask byte-identical; max error ≤ bound.

## Scenario 3 — Invalid bounds rejected; zero bound accepted

Proves FR-008/FR-017/SC-006.

```python
import hoaps_compressor, pytest

for bad in (-0.1, float("inf"), float("nan")):
    try:
        hoaps_compressor.HoapsWvpaCodec(shape=(1, 4, 4), error_bound=bad)
        raise AssertionError("should have raised")
    except ValueError:
        pass

tight = hoaps_compressor.HoapsWvpaCodec(shape=(1, 4, 4), error_bound=0.0)  # valid: tightest bound
```

**Expected**: negative/non-finite bounds raise `ValueError`; `error_bound=0.0` constructs successfully.

## Scenario 4 — CR monotonicity + space-time beats spatial-only

Proves SC-004 and SC-007 on the synthetic suite.

```bash
pytest tests/integration -v
```

Integration suite asserts:

- max abs error ≤ bound for every bound in a sweep (e.g. 0.01/0.05/0.2) — SC-001;
- mask identity in all cases incl. all-missing and no-missing fields — SC-002 (FR-009/FR-010);
- `len(enc_loose) ≤ len(enc_tight)` — SC-004;
- space-time codec CR ≥ spatial-only (per-time-slice) codec CR at equal bound — SC-007;
- decompression returns a float32 array of the configured shape.

## Scenario 5 — Empty / all-missing inputs

Proves FR-009/FR-010.

```python
all_missing = np.full((2, 8, 8), np.nan, dtype="float32")
enc = codec_all_missing.encode(all_missing)
dec = codec_all_missing.decode(enc)
assert np.isnan(dec).all() and enc  # non-empty container, mask restored
```

## Scenario 6 — Attention-enhanced prediction raises CR (enhancement)

Proves the intensive transformer/attention predictor (plan.md "Phase 2 (Enhancement)", research.md R11) raises compression ratio at the same error bound without breaking guarantees.

```bash
# Train the prior on HOAPS data (self-supervised masked-cell prediction)
python scripts/train_prior.py --input data/wvpa_2020-08-01_07.npy --out src/hoaps_compressor/model/weights/prior_v3.hwpm

# Benchmark before/after on the same field and bound
python scripts/compress_stats.py --input data/wvpa_2020-08-01_07.npy --bound 0.05
```

**Expected**:
- `MODEL_VERSION` bumped (container header reflects it); decode of old streams with a different model version raises `ValueError: model version mismatch`.
- Compression ratio at the same bound is strictly higher than the pre-enhancement baseline (recorded in `docs/performance.md`).
- Max error still ≤ bound; mask still bit-identical; encode/decode still deterministic (identical causal walk).

```bash
pytest tests -q   # full suite still green with the new predictor active
```

**Expected**: all existing tests pass (hard error bound, mask fidelity, determinism, config, container) plus the new block-local causal predictor unit tests.

**Expected**: no crash; valid container; mask restored exactly.

## Verifying the guarantees directly

- **Bound**: `np.max(np.abs((dec - orig)[valid])) <= error_bound` — must always hold (FR-003/FR-016).
- **Mask**: `np.array_equal(np.isnan(dec), np.isnan(orig))` (or sentinel comparison, per config) — must always hold (FR-004/FR-015).
- **Metrics**: each container records `max_abs_error`, `n_repaired`, sizes and CR in its header-extra JSON (see [contracts/codec-api.md](contracts/codec-api.md) §5) — inspectable per run (FR-012).
