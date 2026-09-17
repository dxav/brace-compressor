"""Optional Rust acceleration for the BRACE scan."""

try:
    from ._core import (
        causal_scan_decode,
        causal_scan_decode_f64,
        causal_scan_encode,
        causal_scan_encode_f64,
    )
    __all__ = [
        "causal_scan_decode",
        "causal_scan_decode_f64",
        "causal_scan_encode",
        "causal_scan_encode_f64",
    ]
except ImportError:  # The Python codec remains usable without the extension.
    __all__ = []
