"""HOAPS compressor: error-bounded numcodecs codec for wvpa data."""

from .codec import HoapsWvpaCodec

__all__ = ["HoapsWvpaCodec"]

__version__ = "0.1.0"


def _register() -> None:
    """Register the codec with the numcodecs registry (idempotent)."""
    try:
        import numcodecs.registry

        numcodecs.registry.register_codec(HoapsWvpaCodec)
    except Exception:  # pragma: no cover - registration must never break import
        pass


_register()
