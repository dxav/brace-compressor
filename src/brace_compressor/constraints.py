"""Evaluate normalized compression recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .recommendations import RequirementNode


@dataclass(frozen=True, slots=True)
class RequirementCheck:
    """Result of evaluating one requirement and its children."""

    kind: str
    passed: bool
    metric: float | None = None
    limit: float | None = None
    violations: int = 0
    children: tuple["RequirementCheck", ...] = ()

    def get_config(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics."""

        config: dict[str, Any] = {
            "kind": self.kind,
            "passed": self.passed,
            "violations": self.violations,
        }
        if self.metric is not None:
            config["metric"] = self.metric
        if self.limit is not None:
            config["limit"] = self.limit
        if self.children:
            config["children"] = [child.get_config() for child in self.children]
        return config


def check_requirement(
    requirement: RequirementNode,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> RequirementCheck:
    """Evaluate one normalized requirement against two arrays."""

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch between original {original.shape} and "
            f"reconstructed {reconstructed.shape}"
        )
    if requirement.kind in {"any", "all"}:
        children = tuple(
            check_requirement(child, original, reconstructed)
            for child in requirement.children
        )
        passed = (
            any(child.passed for child in children)
            if requirement.kind == "any"
            else all(child.passed for child in children)
        )
        return RequirementCheck(
            kind=requirement.kind,
            passed=passed,
            violations=sum(child.violations for child in children),
            children=children,
        )

    if requirement.kind == "max-pointwise-absolute-error-bound":
        return _pointwise(original, reconstructed, float(requirement.value), relative=False)
    if requirement.kind == "mean-absolute-error-bound":
        return _mean_absolute(original, reconstructed, float(requirement.value))
    if requirement.kind == "max-pointwise-relative-error-bound":
        return _pointwise(original, reconstructed, float(requirement.value), relative=True)
    if requirement.kind == "mean-relative-error-bound":
        return _mean_relative(original, reconstructed, float(requirement.value))
    if requirement.kind == "max-pointwise-range-relative-error-bound":
        return _range_relative(original, reconstructed, float(requirement.value), mean=False)
    if requirement.kind == "mean-range-relative-error-bound":
        return _range_relative(original, reconstructed, float(requirement.value), mean=True)
    if requirement.kind == "max-pointwise-quadratic-error-bound":
        return _quadratic(
            original,
            reconstructed,
            float(requirement.value),
            float(requirement.minimum),
            float(requirement.maximum),
        )
    if requirement.kind == "data-limits":
        return _data_limits(original, reconstructed, requirement.minimum, requirement.maximum)
    if requirement.kind == "isovalue":
        return _isovalue(original, reconstructed, requirement.value)
    if requirement.kind == "missing-value":
        return _missing_value(original, reconstructed, requirement.value)
    if requirement.kind == "lossless":
        passed = (
            np.asarray(original).dtype == np.asarray(reconstructed).dtype
            and np.ascontiguousarray(original).tobytes()
            == np.ascontiguousarray(reconstructed).tobytes()
        )
        return RequirementCheck(kind=requirement.kind, passed=passed, violations=0 if passed else 1)
    raise ValueError(f"unsupported requirement kind {requirement.kind!r}")


def constraint_error_bounds(
    requirement: RequirementNode,
    original: np.ndarray,
) -> np.ndarray:
    """Return per-element absolute tolerances for exact-value constraints."""

    original = np.asarray(original)
    if requirement.kind == "all":
        bounds = [constraint_error_bounds(child, original) for child in requirement.children]
        return np.minimum.reduce(bounds) if bounds else np.full(original.shape, np.inf)
    if requirement.kind == "any":
        raise ValueError("cannot compile an unresolved any requirement")
    if requirement.kind == "data-limits":
        finite = np.isfinite(original)
        applicable = finite.copy()
        if requirement.minimum is not None:
            applicable &= original >= requirement.minimum
        if requirement.maximum is not None:
            applicable &= original <= requirement.maximum
        bounds = np.full(original.shape, np.inf, dtype=np.float64)
        if requirement.minimum is not None:
            bounds[applicable] = np.minimum(
                bounds[applicable], original[applicable] - requirement.minimum
            )
        if requirement.maximum is not None:
            bounds[applicable] = np.minimum(
                bounds[applicable], requirement.maximum - original[applicable]
            )
        return bounds
    if requirement.kind == "max-pointwise-absolute-error-bound":
        return np.full(original.shape, float(requirement.value), dtype=np.float64)
    if requirement.kind == "max-pointwise-relative-error-bound":
        return np.abs(original.astype(np.float64)) * float(requirement.value)
    if requirement.kind == "max-pointwise-range-relative-error-bound":
        finite = original[np.isfinite(original)]
        value_range = float(np.ptp(finite)) if finite.size else 0.0
        return np.full(original.shape, value_range * float(requirement.value), dtype=np.float64)
    if requirement.kind == "isovalue":
        value = float(requirement.value)
        bounds = np.full(original.shape, np.inf, dtype=np.float64)
        finite = np.isfinite(original)
        bounds[finite] = np.abs(original[finite] - value)
        return bounds
    if requirement.kind == "max-pointwise-quadratic-error-bound":
        value = float(requirement.value)
        minimum = float(requirement.minimum)
        maximum = float(requirement.maximum)
        bounds = np.zeros(original.shape, dtype=np.float64)
        if maximum <= minimum:
            return bounds
        finite = np.isfinite(original)
        inside = finite & (original > minimum) & (original < maximum)
        scale = 1.0 - (
            2.0 * (original.astype(np.float64) - minimum) / (maximum - minimum) - 1.0
        ) ** 2
        bounds[inside] = scale[inside] * value
        return bounds
    return np.full(original.shape, np.inf, dtype=np.float64)


def _special_equal(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    return (np.isnan(original) & np.isnan(reconstructed)) | (original == reconstructed)


def _finite_error(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original)
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.abs(reconstructed.astype(np.float64) - original.astype(np.float64))
    return finite, errors


def _pointwise(original, reconstructed, bound: float, *, relative: bool) -> RequirementCheck:
    finite, errors = _finite_error(original, reconstructed)
    exact_special = _special_equal(original, reconstructed)
    if relative:
        zero = finite & (original == 0)
        limits = np.abs(original.astype(np.float64)) * bound
        allowed = finite & ~zero & (errors <= limits)
        valid = (finite & ~zero & allowed) | (zero & exact_special) | (~finite & exact_special)
    else:
        allowed = finite & (errors <= bound)
        valid = allowed | (~finite & exact_special)
    failed = ~valid
    return RequirementCheck(
        kind="max-pointwise-relative-error-bound" if relative else "max-pointwise-absolute-error-bound",
        passed=not np.any(failed),
        metric=float(errors[finite].max()) if np.any(finite) else 0.0,
        limit=bound,
        violations=int(failed.sum()),
    )


def _mean_absolute(original, reconstructed, bound: float) -> RequirementCheck:
    finite, errors = _finite_error(original, reconstructed)
    special_ok = _special_equal(original, reconstructed)
    special_failed = (~finite) & ~special_ok
    metric = float(errors[finite].mean()) if np.any(finite) else 0.0
    passed = not np.any(special_failed) and metric <= bound
    return RequirementCheck("mean-absolute-error-bound", passed, metric, bound, int(special_failed.sum()))


def _mean_relative(original, reconstructed, bound: float) -> RequirementCheck:
    finite, errors = _finite_error(original, reconstructed)
    special_ok = _special_equal(original, reconstructed)
    nonzero = finite & (original != 0)
    denominator = np.abs(original.astype(np.float64))
    numerator = float(errors[finite].sum())
    limit = float((denominator[finite] * bound).sum())
    zero_failed = finite & (original == 0) & ~special_ok
    special_failed = (~finite) & ~special_ok
    passed = not np.any(zero_failed | special_failed) and numerator <= limit
    metric = numerator / float(denominator[finite].sum()) if np.any(nonzero) else 0.0
    return RequirementCheck("mean-relative-error-bound", passed, metric, bound, int((zero_failed | special_failed).sum()))


def _range_relative(original, reconstructed, bound: float, *, mean: bool) -> RequirementCheck:
    finite, errors = _finite_error(original, reconstructed)
    special_ok = _special_equal(original, reconstructed)
    finite_values = original[finite].astype(np.float64)
    value_range = float(np.ptp(finite_values)) if finite_values.size else 0.0
    finite_metric = float(errors[finite].mean()) if mean and np.any(finite) else float(errors[finite].max()) if np.any(finite) else 0.0
    limit = value_range * bound
    special_failed = (~finite) & ~special_ok
    passed = not np.any(special_failed) and (
        finite_metric <= limit if value_range > 0.0 else np.all(errors[finite] == 0.0)
    )
    kind = "mean-range-relative-error-bound" if mean else "max-pointwise-range-relative-error-bound"
    return RequirementCheck(kind, passed, finite_metric, limit, int(special_failed.sum()))


def _quadratic(original, reconstructed, bound: float, minimum: float, maximum: float) -> RequirementCheck:
    finite, errors = _finite_error(original, reconstructed)
    special_ok = _special_equal(original, reconstructed)
    inside = finite & (original > minimum) & (original < maximum) & (maximum > minimum)
    scale = 1.0 - (2.0 * (original.astype(np.float64) - minimum) / (maximum - minimum) - 1.0) ** 2
    allowed = inside & (errors <= scale * bound)
    exact = (~inside) & special_ok
    valid = allowed | exact | ((~finite) & special_ok)
    failed = ~valid
    return RequirementCheck(
        "max-pointwise-quadratic-error-bound",
        not np.any(failed),
        float(errors[finite].max()) if np.any(finite) else 0.0,
        bound,
        int(failed.sum()),
    )


def _data_limits(original, reconstructed, minimum, maximum) -> RequirementCheck:
    finite = np.isfinite(original)
    applicable = finite.copy()
    if minimum is not None:
        applicable &= original >= minimum
    if maximum is not None:
        applicable &= original <= maximum
    valid = ~applicable | (~finite) | (
        (minimum is None or reconstructed >= minimum)
        & (maximum is None or reconstructed <= maximum)
    )
    failed = ~valid
    return RequirementCheck("data-limits", not np.any(failed), None, None, int(failed.sum()))


def _isovalue(original, reconstructed, value) -> RequirementCheck:
    original_relation = np.sign(original - value)
    reconstructed_relation = np.sign(reconstructed - value)
    valid = original_relation == reconstructed_relation
    valid |= _special_equal(original, reconstructed)
    failed = ~valid
    return RequirementCheck("isovalue", not np.any(failed), None, None, int(failed.sum()))


def _missing_value(original, reconstructed, value) -> RequirementCheck:
    selected = np.isnan(original) if isinstance(value, float) and np.isnan(value) else original == value
    valid = ~selected | _special_equal(original, reconstructed)
    failed = ~valid
    return RequirementCheck("missing-value", not np.any(failed), None, None, int(failed.sum()))