"""Tests for typed compression-recommendation integration."""

import pytest
import numpy as np
from compression_recommendations import Recommendations

from brace_compressor.recommendations import (
    ErrorBoundRecommendation,
    plan_recommendation,
    recommend_error_bound,
)
from brace_compressor import BraceCodec
from brace_compressor.container import read_container


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


def test_codec_factory_conservatively_supports_mean_absolute_requirement():
    codec = BraceCodec.from_recommendation(
        shape=(1, 1, 1), variable="tp", level_kind="single", dtype="float64"
    )

    assert codec.error_bound == 1e-5
    assert codec.error_bound_mode == "absolute"


def test_codec_factory_records_plan_and_constraint_diagnostics():
    codec = BraceCodec.from_recommendation(
        shape=(1, 1, 2), variable="cc", dtype="float64"
    )

    encoded = codec.encode(np.array([[[1.0, 2.0]]], dtype=np.float64))
    header = read_container(encoded).header_extra

    assert header["recommendation_plan"]["variable"] == "cc"
    assert header["recommendation_checks"][0]["passed"]


def test_range_relative_bound_is_resolved_from_source_data():
    recommendations = Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": [
                    {
                        "kind": "max-pointwise-range-relative-error-bound",
                        "value": 0.1,
                    }
                ],
            }
        ],
        version="0.1.0",
        metadata={},
    )
    codec = BraceCodec.from_recommendation(
        shape=(1, 1, 2),
        variable="x",
        dtype="float64",
        recommendations=recommendations,
    )

    header = read_container(codec.encode(np.array([[[1.0, 3.0]]]))).header_extra

    assert header["error_bound"] == pytest.approx(0.2)
    assert header["recommendation_checks"][0]["passed"]


def test_plan_preserves_all_requirements_and_selects_relative_any_branch():
    plan = plan_recommendation("pv")

    assert plan.requirements[0].kind == "any"
    assert plan.requirements[0].children[0].children[0].kind == (
        "max-pointwise-absolute-error-bound"
    )
    assert plan.selected[0].kind == "max-pointwise-relative-error-bound"
    assert plan.selected[0].value == 0.1


def test_plan_preserves_mean_requirement_tree():
    plan = plan_recommendation("tp", level_kind="single")

    assert plan.requirements[0].kind == "any"
    assert plan.selected[0].kind == "mean-absolute-error-bound"
    assert plan.selected[0].value == 1e-5