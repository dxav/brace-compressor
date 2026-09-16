"""BRACE: bounded residual adaptive compression for wvpa data."""

from .codec import BraceCodec

__all__ = ["BraceCodec"]

__version__ = "0.1.0"


def _register() -> None:
    """Register the codec with the numcodecs registry (idempotent)."""
    try:
        import numcodecs.registry

        numcodecs.registry.register_codec(BraceCodec)
    except Exception:  # pragma: no cover - registration must never break import
        pass


_register()
