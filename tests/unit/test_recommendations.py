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


def make_recommendations(requirements):
    return Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": requirements,
            }
        ],
        version="0.1.0",
        metadata={},
    )


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


def test_data_limits_recommendation_is_verified_after_encoding():
    recommendations = Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": [
                    {"kind": "data-limits", "minimum": 0.0, "maximum": 1.0}
                ],
            }
        ],
        version="0.1.0",
        metadata={},
    )
    original = np.array([[[-1.0, 0.5, 2.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=recommendations,
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)

    assert np.array_equal(decoded, original)
    assert read_container(encoded).header_extra["recommendation_checks"][0]["passed"]


def test_data_limits_can_use_explicit_base_bound_for_unconstrained_values():
    recommendations = make_recommendations(
        [{"kind": "data-limits", "minimum": 0.0, "maximum": 1.0}]
    )
    original = np.linspace(-2.0, 3.0, 32, dtype=np.float64).reshape(1, 1, 32)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        error_bound=0.1,
        recommendations=recommendations,
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert header["recommendation_checks"][0]["passed"]
    assert header["n_repaired"] < original.size
    assert np.any(decoded != original)


def test_isovalue_recommendation_preserves_threshold_classification():
    recommendations = Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": [{"kind": "isovalue", "value": 0.5}],
            }
        ],
        version="0.1.0",
        metadata={},
    )
    original = np.array([[[-1.0, 0.5, 2.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=recommendations,
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)

    assert np.sign(decoded - 0.5).tolist() == np.sign(original - 0.5).tolist()
    assert read_container(encoded).header_extra["recommendation_checks"][0]["passed"]


def test_missing_value_recommendation_binds_codec_sentinel():
    recommendations = Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": [{"kind": "missing-value", "value": -999.0}],
            }
        ],
        version="0.1.0",
        metadata={},
    )
    original = np.array([[[-999.0, 1.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=recommendations,
    )

    decoded = codec.decode(codec.encode(original))

    assert codec.missing_value == -999.0
    assert np.array_equal(decoded, original)


def test_lossless_recommendation_preserves_configured_dtype_bits():
    recommendations = Recommendations.from_config(
        recommendations=[
            {
                "filters": [
                    {"kind": "cf-short-name", "value": "x"},
                    {"kind": "level-kind", "value": "pressure"},
                ],
                "requirements": [{"kind": "lossless"}],
            }
        ],
        version="0.1.0",
        metadata={},
    )
    bits = np.array([0x8000000000000000, 0x7FF8000000000042], dtype=np.uint64)
    original = bits.view(np.float64).reshape(1, 1, 2)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=recommendations,
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)

    assert decoded.tobytes() == original.tobytes()
    assert read_container(encoded).header_extra["lossless_transform"] == "byte-shuffle"


def test_mean_relative_recommendation_is_verified_after_encoding():
    original = np.array([[[1.0, 2.0, 4.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [{"kind": "mean-relative-error-bound", "value": 0.1}]
        ),
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert decoded.shape == original.shape
    assert header["recommendation_checks"][0]["passed"]


def test_mean_range_relative_recommendation_is_verified_after_encoding():
    original = np.array([[[0.0, 5.0, 10.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [{"kind": "mean-range-relative-error-bound", "value": 0.05}]
        ),
    )

    encoded = codec.encode(original)
    codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert header["error_bound"] == pytest.approx(0.5)
    assert header["recommendation_checks"][0]["passed"]


def test_quadratic_recommendation_is_verified_after_encoding():
    original = np.array([[[0.0, 5.0, 10.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [
                {
                    "kind": "max-pointwise-quadratic-error-bound",
                    "value": 0.5,
                    "minimum": 0.0,
                    "maximum": 10.0,
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert header["recommendation_checks"][0]["passed"]
    assert header["block_size"] == 256
    assert len(header["block_steps"]) == 1


def test_quadratic_policy_does_not_force_unrelated_values_to_exact_repairs():
    original = np.linspace(0.0, 10.0, 32, dtype=np.float64).reshape(1, 1, 32)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [
                {
                    "kind": "max-pointwise-quadratic-error-bound",
                    "value": 0.5,
                    "minimum": 0.0,
                    "maximum": 10.0,
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert header["recommendation_checks"][0]["passed"]
    assert header["n_repaired"] < original.size
    assert np.any(decoded != original)


def test_quadratic_local_steps_round_trip_multiple_blocks():
    original = np.linspace(0.0, 10.0, 513, dtype=np.float64).reshape(1, 1, 513)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [
                {
                    "kind": "max-pointwise-quadratic-error-bound",
                    "value": 0.5,
                    "minimum": 0.0,
                    "maximum": 10.0,
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    decoded = codec.decode(encoded)
    header = read_container(encoded).header_extra

    assert len(header["block_steps"]) == 3
    assert header["recommendation_checks"][0]["passed"]
    assert header["n_repaired"] < original.size
    assert decoded.shape == original.shape


def test_all_recommendation_constraints_are_verified_together():
    original = np.array([[[-1.0, 0.5, 2.0]]], dtype=np.float64)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [
                {
                    "kind": "all",
                    "requirements": [
                        {"kind": "max-pointwise-absolute-error-bound", "value": 0.1},
                        {"kind": "data-limits", "minimum": 0.0, "maximum": 1.0},
                    ],
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    codec.decode(encoded)
    check = read_container(encoded).header_extra["recommendation_checks"][0]

    assert check["kind"] == "all"
    assert check["passed"]
    assert all(child["passed"] for child in check["children"])


def test_all_mean_and_pointwise_constraints_both_remain_enforced():
    original = np.linspace(-100.0, 100.0, 513, dtype=np.float64).reshape(1, 1, 513)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float64",
        recommendations=make_recommendations(
            [
                {
                    "kind": "all",
                    "requirements": [
                        {"kind": "mean-absolute-error-bound", "value": 5.0},
                        {"kind": "max-pointwise-absolute-error-bound", "value": 10.0},
                    ],
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    check = read_container(encoded).header_extra["recommendation_checks"][0]
    children = {child["kind"]: child for child in check["children"]}

    assert check["passed"]
    assert children["mean-absolute-error-bound"]["passed"]
    assert children["max-pointwise-absolute-error-bound"]["passed"]
    assert children["max-pointwise-absolute-error-bound"]["metric"] <= 10.0


def test_any_recommendation_selects_one_supported_branch():
    plan = plan_recommendation(
        "x",
        recommendations=make_recommendations(
            [
                {
                    "kind": "any",
                    "requirements": [
                        {"kind": "mean-relative-error-bound", "value": 0.1},
                        {
                            "kind": "max-pointwise-quadratic-error-bound",
                            "value": 0.5,
                            "minimum": 0.0,
                            "maximum": 10.0,
                        },
                    ],
                }
            ]
        ),
    )

    assert len(plan.selected) == 1
    assert plan.selected[0].kind == "mean-relative-error-bound"


def test_any_recommendation_uses_data_scale_for_encoding_branch():
    original = np.array([[[1.0, 1.5, 2.0]]], dtype=np.float32)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float32",
        recommendations=make_recommendations(
            [
                {
                    "kind": "any",
                    "requirements": [
                        {"kind": "max-pointwise-relative-error-bound", "value": 0.1},
                        {"kind": "max-pointwise-absolute-error-bound", "value": 2.0},
                    ],
                }
            ]
        ),
    )

    encoded = codec.encode(original)
    selected = read_container(encoded).header_extra["recommendation_plan"]["selected"]

    assert selected[0]["kind"] == "max-pointwise-absolute-error-bound"


def test_data_selected_any_branch_updates_codec_mode_and_bound():
    original = np.array([[[1.0, 1.5, 2.0]]], dtype=np.float32)
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable="x",
        dtype="float32",
        recommendations=make_recommendations(
            [
                {
                    "kind": "any",
                    "requirements": [
                        {"kind": "max-pointwise-relative-error-bound", "value": 0.1},
                        {"kind": "max-pointwise-absolute-error-bound", "value": 2.0},
                    ],
                }
            ]
        ),
    )

    header = read_container(codec.encode(original)).header_extra

    assert header["error_bound_mode"] == "absolute"
    assert header["error_bound"] == pytest.approx(2.0)
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