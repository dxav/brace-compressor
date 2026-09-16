"""Error-bound validation (FR-008, FR-017).

A bound must be a real, finite number >= 0. Zero is valid and selects the
lossless raw-float32 path for valid values.
"""

from __future__ import annotations

import math
from typing import Union

Bound = Union[int, float]


def validate_error_bound(value: Bound) -> float:
    """Validate and normalize an absolute error bound.

    Raises:
        ValueError: if ``value`` is negative or non-finite (FR-008).
        TypeError: if ``value`` is not a real number.

    Returns:
        float: the validated bound. ``0.0`` is legal and selects the exact
        raw-float32 path for valid values.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"error_bound must be a real number, got {type(value).__name__}"
        )
    bound = float(value)
    if math.isnan(bound) or math.isinf(bound):
        raise ValueError(
            "error_bound must be finite; got non-finite value "
            f"({value!r}). Negative and non-finite bounds are rejected "
            "per FR-008; 0.0 is accepted as the tightest allowed bound."
        )
    if bound < 0.0:
        raise ValueError(
            f"error_bound must be >= 0; got {bound!r}. Negative bounds "
            "are rejected per FR-008; 0.0 is accepted as the tightest "
            "allowed bound and selects the exact raw-float32 path."
        )
    return bound


def normalize_missing_value(missing_value: Union[float, str]) -> float:
    """Normalize the missing sentinel: ``"nan"`` -> ``nan``; finite floats pass through.

    Raises:
        ValueError: on non-finite float sentinels (use the string ``"nan"``).
        TypeError: on non-numeric, non-``"nan"`` values.
    """
    if isinstance(missing_value, str):
        if missing_value.lower() == "nan":
            return float("nan")
        raise ValueError(
            "missing_value string must be \"nan\"; got "
            f"{missing_value!r}. Pass a finite float or \"nan\"."
        )
    if isinstance(missing_value, bool) or not isinstance(missing_value, (int, float)):
        raise TypeError(
            "missing_value must be a finite float or the string \"nan\"; "
            f"got {type(missing_value).__name__}"
        )
    sentinel = float(missing_value)
    if math.isnan(sentinel):
        # Treat NaN configs as the canonical "nan" sentinel
        return float("nan")
    if math.isinf(sentinel):
        raise ValueError(
            "missing_value must be finite; use the string \"nan\" for "
            "NaN-as-missing."
        )
    return sentinel


def validate_shape(shape) -> tuple[int, int, int]:
    """Validate a 2-D or 3-D grid shape and normalize it to 3-D.

    A 2-D ``(lat, lon)`` field is represented internally as one time slice.
    """
    try:
        dims = tuple(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"shape must be a 2-tuple (lat, lon) or 3-tuple "
            f"(time, lat, lon); got {shape!r}"
        ) from exc
    if len(dims) == 2:
        dims = (1, *dims)
    elif len(dims) != 3:
        raise ValueError(
            f"shape must be a 2-tuple (lat, lon) or 3-tuple "
            f"(time, lat, lon); got {shape!r}"
        )
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
            raise ValueError(
                f"shape components must be positive integers; got {shape!r}"
            )
    return tuple(int(dimension) for dimension in dims)
