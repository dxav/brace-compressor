"""BRACE: bounded residual adaptive compression for gridded data."""

from .codec import BraceCodec
from .recommendations import ErrorBoundRecommendation, load_recommendations
from .recommendations import recommend_error_bound

__all__ = [
    "BraceCodec",
    "ErrorBoundRecommendation",
    "load_recommendations",
    "recommend_error_bound",
]

__version__ = "0.1.0"


def _register() -> None:
    """Register the codec with the numcodecs registry (idempotent)."""
    try:
        import numcodecs.registry

        numcodecs.registry.register_codec(BraceCodec)
    except Exception:  # pragma: no cover - registration must never break import
        pass


_register()
