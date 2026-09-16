# hoaps-compressor

Error-bounded, causal-predictor `numcodecs` codec for HOAPS water-vapor
(`wvpa`) gridded climate fields.

## Guarantees

- **Absolute error bound** (FR-003/FR-016): every reconstructed value is
  within the user-specified bound of the original — verified
  encode-side before any stream is returned. At `error_bound=0` the
  guarantee is exempted per FR-003 (tightest available representation).
- **Missing values preserved bit-exactly** (FR-004/FR-015): a losslessly
  stored bitmask restores the sentinel exactly; no valid↔missing flips.
- **Space-time causal prediction**: reconstructed spatial and temporal
  neighbors are used symmetrically by encode and decode.

## Install / test

```bash
pip install -e ".[test]"
pytest tests -q              # full suite (slow integration ~10 min)
pytest tests/unit -q         # fast subset
```

## Usage

```python
import numpy as np
import hoaps_compressor  # registers "hoaps-wvpa" with numcodecs
from hoaps_compressor import HoapsWvpaCodec

shape = (8, 90, 180)
field = ...  # float32 (time, lat, lon); NaN = missing
codec = HoapsWvpaCodec(shape=shape, error_bound=0.05)
enc = codec.encode(field)         # bytes
dec = codec.decode(enc)           # float32 array, same shape

# numcodecs registry / config round trip
cfg = codec.get_config()          # JSON-serializable, id="hoaps-wvpa"
codec2 = hoaps_compressor.HoapsWvpaCodec.from_config(cfg)
```

### Statistics utility

Run a compression round trip with a full CR/statistics report on a
synthetic HOAPS-like field (or your own `.npy`):

```bash
.venv/bin/python scripts/compress_stats.py --shape 8 90 180 --bound 0.05
.venv/bin/python scripts/compress_stats.py --input myfield.npy --bound 0.01
.venv/bin/python scripts/compress_stats.py --shape 4 32 64 --sweep 0.01 0.05 0.2  # CR-vs-accuracy table
.venv/bin/python scripts/compress_stats.py --shape 2 16 16 --json                 # machine-readable output
```

## Design docs

- **Architecture** (pipeline, causal predictor, and entropy coding): `docs/architecture.md`
- Spec: `specs/001-hoaps-wvpa-compressor/spec.md`
- Plan/research: `specs/001-hoaps-wvpa-compressor/plan.md`, `research.md`
- Contract: `specs/001-hoaps-wvpa-compressor/contracts/codec-api.md`
- Performance: `docs/performance.md`
