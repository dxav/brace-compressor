"""Optional Rust acceleration for the BRACE scan."""

try:
    from ._core import causal_scan_decode, causal_scan_encode
    __all__ = ["causal_scan_decode", "causal_scan_encode"]
except ImportError:  # The Python codec remains usable without the extension.
    __all__ = []
