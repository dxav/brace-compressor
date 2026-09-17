"""Optional Rust acceleration for the BRACE scan."""

try:
    from . import _core
except ImportError:  # The Python codec remains usable without the extension.
    _core = None

__all__ = []
if _core is not None:
    for _name in (
        "causal_scan_decode",
        "causal_scan_decode_f64",
        "causal_scan_encode",
        "causal_scan_encode_f64",
        "ctx_rans_decode",
        "ctx_rans_encode",
        "rans_decode",
        "rans_encode",
    ):
        if hasattr(_core, _name):
            globals()[_name] = getattr(_core, _name)
            __all__.append(_name)
