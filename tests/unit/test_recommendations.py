"""Tests for typed compression-recommendation integration."""

import pytest

from brace_compressor.recommendations import (
    ErrorBoundRecommendation,
    recommend_error_bound,
)
from brace_compressor import BraceCodec


def test_relative_recommendation_is_extracted_from_typed_model():
    recommendation = recommend_error_bound("cc")

    assert recommendation == ErrorBoundRecommendation(mode="relative", value=0.01)


def test_absolute_recommendation_is_extracted_from_typed_model():
    recommendation = recommend_error_bound("u")

    assert recommendation == ErrorBoundRecommendation(mode="absolute", value=0.5)


def test_relative_bound_is_preferred_when_alternatives_include_both_modes():
    recommendation = recommend_error_bound("pv")

    assert recommendation == ErrorBoundRecommendation(mode="relative", value=0.1)


def test_absolute_bound_is_used_when_relative_requirement_is_not_pointwise():
    recommendation = recommend_error_bound("t")

    assert recommendation == ErrorBoundRecommendation(mode="absolute", value=0.05)


def test_unknown_variable_reports_missing_recommendation():
    with pytest.raises(KeyError, match="failed to find a compatible error bound"):
        recommend_error_bound("does_not_exist")


def test_codec_factory_applies_recommendation():
    codec = BraceCodec.from_recommendation(
        shape=(1, 1, 1), variable="cc", dtype="float64"
    )

    assert codec.error_bound == 0.01
    assert codec.error_bound_mode == "relative"