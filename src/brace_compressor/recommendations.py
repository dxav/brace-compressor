"""Extract codec-compatible error bounds from typed recommendations."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from compression_recommendations import Recommendations

ErrorBoundMode = Literal["absolute", "relative"]


@dataclass(frozen=True, slots=True)
class ErrorBoundRecommendation:
    """A pointwise error bound that can be applied by :class:`BraceCodec`."""

    mode: ErrorBoundMode
    value: float


def load_recommendations() -> Recommendations:
    """Load the package-provided, strongly typed recommendations."""

    return Recommendations.provide


def recommend_error_bound(
    variable: str,
    *,
    level_kind: str = "pressure",
    markers: Mapping[str, None | bool | int | float | str] | None = None,
    recommendations: Recommendations | None = None,
) -> ErrorBoundRecommendation:
    """Select a codec-compatible pointwise bound for a variable.

    Pointwise-relative bounds are preferred when recommendations offer both
    modes. Unsupported requirements, such as range-relative bounds, are not
    reinterpreted; an available pointwise absolute bound is used instead.
    """

    search_markers: dict[str, None | bool | int | float | str] = {
        "cf-short-name": variable,
        "grib-short-name": variable,
        "level-kind": level_kind,
    }
    if markers is not None:
        search_markers.update(markers)
    source = recommendations or load_recommendations()
    try:
        requirements = source.search(markers=search_markers)
    except KeyError as exc:
        raise KeyError("failed to find a compatible error bound", search_markers) from exc

    candidates = tuple(_iter_bound_candidates(requirements))
    if not candidates:
        raise KeyError("failed to find a compatible error bound", search_markers)
    relative = tuple(value for mode, value in candidates if mode == "relative")
    if relative:
        return ErrorBoundRecommendation(mode="relative", value=min(relative))
    absolute = tuple(value for mode, value in candidates if mode == "absolute")
    return ErrorBoundRecommendation(mode="absolute", value=min(absolute))


def _iter_bound_candidates(requirements: Collection[object]):
    for requirement in requirements:
        children = getattr(requirement, "requirements", None)
        if children is not None:
            yield from _iter_bound_candidates(children)
            continue
        kind = getattr(getattr(requirement, "kind", None), "value", None)
        mode = {
            "max-pointwise-absolute-error-bound": "absolute",
            "max-pointwise-relative-error-bound": "relative",
        }.get(kind)
        value = getattr(requirement, "value", None)
        if mode is not None and isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value) and value >= 0.0:
                yield mode, value