"""Tests for recommendation constraint evaluation."""

import numpy as np

from brace_compressor.constraints import check_requirement
from brace_compressor.recommendations import RequirementNode


def check(kind, original, reconstructed, **kwargs):
    return check_requirement(
        RequirementNode(kind=kind, **kwargs),
        np.asarray(original, dtype=np.float64),
        np.asarray(reconstructed, dtype=np.float64),
    )


def test_pointwise_absolute_and_relative_bounds():
    original = [1.0, 2.0, 0.0, np.nan, np.inf]
    reconstructed = [1.09, 2.09, 0.0, np.nan, np.inf]

    assert check("max-pointwise-absolute-error-bound", original, reconstructed, value=0.1).passed
    assert check("max-pointwise-relative-error-bound", original, reconstructed, value=0.1).passed


def test_mean_bounds_are_global_constraints():
    original = [1.0, 2.0, 4.0]
    reconstructed = [1.1, 2.0, 3.9]

    assert check("mean-absolute-error-bound", original, reconstructed, value=0.1).passed
    assert check("mean-relative-error-bound", original, reconstructed, value=0.05).passed


def test_range_relative_and_quadratic_bounds():
    original = [0.0, 5.0, 10.0]
    reconstructed = [0.5, 5.0, 9.5]

    assert check(
        "max-pointwise-range-relative-error-bound", original, reconstructed, value=0.05
    ).passed
    assert check(
        "mean-range-relative-error-bound", original, reconstructed, value=0.05
    ).passed
    assert check(
        "max-pointwise-quadratic-error-bound",
        original,
        [0.0, 5.0, 10.0],
        value=0.5,
        minimum=0.0,
        maximum=10.0,
    ).passed


def test_limits_isovalue_missing_and_lossless_constraints():
    original = [0.0, 1.0, np.nan]
    reconstructed = [0.0, 1.0, np.nan]

    assert check("data-limits", original, reconstructed, minimum=0.0, maximum=1.0).passed
    assert check("isovalue", [0.0, 1.0], [0.0, 1.1], value=0.0).passed
    assert not check("isovalue", [0.0], [1.0], value=1.0).passed
    assert check("missing-value", [np.nan, 1.0], [np.nan, 1.0], value=np.nan).passed
    assert check("lossless", original, reconstructed).passed


def test_any_and_all_use_logical_semantics():
    node = RequirementNode(
        kind="all",
        children=(
            RequirementNode(kind="any", children=(
                RequirementNode(kind="max-pointwise-absolute-error-bound", value=0.2),
                RequirementNode(kind="max-pointwise-relative-error-bound", value=0.01),
            )),
            RequirementNode(kind="data-limits", minimum=0.0),
        ),
    )

    result = check_requirement(node, np.array([1.0, 2.0]), np.array([1.1, 2.0]))

    assert result.passed
    assert result.children[0].children[0].passed
    assert not result.children[0].children[1].passed