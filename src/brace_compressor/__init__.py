"""BRACE: bounded residual adaptive compression for gridded data."""

from .codec import BraceCodec
from .constraints import RequirementCheck, check_requirement
from .recommendations import (
    ErrorBoundRecommendation,
    RecommendationPlan,
    RequirementNode,
    load_recommendations,
    plan_recommendation,
    recommend_error_bound,
)

__all__ = [
    "BraceCodec",
    "ErrorBoundRecommendation",
    "RecommendationPlan",
    "RequirementCheck",
    "RequirementNode",
    "check_requirement",
    "load_recommendations",
    "plan_recommendation",
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
