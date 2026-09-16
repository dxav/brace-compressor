# Quickstart: HOAPS WVPA Causal Compressor

```bash
pip install -e ".[test]"
pytest tests -q
```

```python
import numpy as np
from hoaps_compressor import HoapsWvpaCodec

shape = (8, 32, 64)
field = np.zeros(shape, dtype="float32")
field[:, :4, :] = np.nan
codec = HoapsWvpaCodec(shape=shape, error_bound=0.05)
encoded = codec.encode(field)
decoded = codec.decode(encoded)
assert np.array_equal(np.isnan(decoded), np.isnan(field))
```

Use `codec.get_config()` and `HoapsWvpaCodec.from_config(config)` for registry/configuration integration. The integration suite covers bound enforcement, all-missing fields, monotonic compression, and the space-time baseline.
