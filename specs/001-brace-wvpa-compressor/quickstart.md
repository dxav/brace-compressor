# Quickstart: BRACE WVPA Codec

```bash
pip install -e ".[test]"
pytest tests -q
```

```python
import numpy as np
from brace_compressor import BraceCodec

shape = (8, 32, 64)
field = np.zeros(shape, dtype="float32")
field[:, :4, :] = np.nan
codec = BraceCodec(shape=shape, error_bound=0.05)
encoded = codec.encode(field)
decoded = codec.decode(encoded)
assert np.array_equal(np.isnan(decoded), np.isnan(field))
```

Use `codec.get_config()` and `BraceCodec.from_config(config)` for registry/configuration integration. The integration suite covers bound enforcement, all-missing fields, monotonic compression, and the space-time baseline.
