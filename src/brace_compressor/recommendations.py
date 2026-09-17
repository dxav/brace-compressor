"""Plan typed compression recommendations for BRACE strategies."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from compression_recommendations import Recommendations

ErrorBoundMode = Literal["absolute", "relative"]

_KIND_PREFERENCE = {
    "max-pointwise-relative-error-bound": 0,
    "max-pointwise-absolute-error-bound": 1,
    "lossless": 2,
    "mean-relative-error-bound": 3,
    "mean-absolute-error-bound": 4,
    "max-pointwise-range-relative-error-bound": 5,
    "mean-range-relative-error-bound": 6,
    "max-pointwise-quadratic-error-bound": 7,
    "isovalue": 8,
    "missing-value": 9,
    "data-limits": 10,
}


@dataclass(frozen=True, slots=True)
class RequirementNode:
    """Canonical, JSON-compatible representation of one requirement node."""

    kind: str
    children: tuple["RequirementNode", ...] = ()
    value: float | int | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None

    def get_config(self) -> dict[str, object]:
        """Return the canonical package-independent representation."""

        config: dict[str, object] = {"kind": self.kind}
        if self.children:
            config["requirements"] = [child.get_config() for child in self.children]
        if self.value is not None:
            config["value"] = self.value
        if self.minimum is not None:
            config["minimum"] = self.minimum
        if self.maximum is not None:
            config["maximum"] = self.maximum
        return config


@dataclass(frozen=True, slots=True)
class RecommendationPlan:
    """A full recommendation tree and the branch selected for encoding."""

    variable: str
    markers: Mapping[str, None | bool | int | float | str]
    recommendations_version: str
    requirements: tuple[RequirementNode, ...]
    selected_tree: tuple[RequirementNode, ...]

    @property
    def selected(self) -> tuple[RequirementNode, ...]:
        """Return selected leaf requirements for strategy planning."""

        return tuple(_leaves(self.selected_tree))

    def get_config(self) -> dict[str, object]:
        """Return JSON-compatible plan metadata."""

        return {
            "variable": self.variable,
            "markers": dict(self.markers),
            "recommendations_version": self.recommendations_version,
            "requirements": [node.get_config() for node in self.requirements],
            "selected": [node.get_config() for node in self.selected_tree],
        }

    def select_for_data(self, data) -> "RecommendationPlan":
        """Select ``any`` branches using the field's admissible error scale."""

        import numpy as np

        values = np.asarray(data)
        return RecommendationPlan(
            variable=self.variable,
            markers=self.markers,
            recommendations_version=self.recommendations_version,
            requirements=self.requirements,
            selected_tree=tuple(
                _select_node(node, values) for node in self.requirements
            ),
        )

    def candidate_plans_for_data(self, data) -> tuple["RecommendationPlan", ...]:
        """Enumerate complete, data-aware candidate plans for ``any`` nodes."""

        import numpy as np

        values = np.asarray(data)
        trees = [tuple([node]) for node in self.requirements]
        for index, node in enumerate(self.requirements):
            choices = _expand_node(node, values)
            trees = [
                tree[:index] + (choice,) + tree[index + 1 :]
                for tree in trees
                for choice in choices
            ]
        return tuple(
            RecommendationPlan(
                variable=self.variable,
                markers=self.markers,
                recommendations_version=self.recommendations_version,
                requirements=self.requirements,
                selected_tree=tree,
            )
            for tree in trees
        )

    def pointwise_error_bound(self) -> ErrorBoundRecommendation:
        """Return a conservative scalar bound for the current codec.

        Mean absolute and mean relative requirements are enforced pointwise by
        BRACE until aggregate-aware encoding strategies are available.
        """

        candidates = tuple(
            (node.kind, float(node.value))
            for node in self.selected
            if node.kind
            in {
                "max-pointwise-absolute-error-bound",
                "max-pointwise-relative-error-bound",
                "mean-absolute-error-bound",
                "mean-relative-error-bound",
            }
            and isinstance(node.value, (int, float))
            and math.isfinite(float(node.value))
            and float(node.value) >= 0.0
        )
        if not candidates:
            raise KeyError(
                "failed to find a compatible pointwise error bound",
                self.markers,
            )
        relative = tuple(
            value
            for kind, value in candidates
            if kind in {
                "max-pointwise-relative-error-bound",
                "mean-relative-error-bound",
            }
        )
        if relative:
            return ErrorBoundRecommendation(mode="relative", value=min(relative))
        return ErrorBoundRecommendation(
            mode="absolute", value=min(value for _, value in candidates)
        )

    def range_relative_error_bound(self, data) -> ErrorBoundRecommendation:
        """Resolve a range-relative requirement into an absolute bound."""

        import numpy as np

        candidates = tuple(
            float(node.value)
            for node in self.selected
            if node.kind
            in {
                "max-pointwise-range-relative-error-bound",
                "mean-range-relative-error-bound",
            }
            and isinstance(node.value, (int, float))
        )
        if not candidates:
            raise KeyError("selected recommendation has no range-relative bound")
        finite = np.asarray(data)[np.isfinite(data)]
        data_range = float(np.ptp(finite)) if finite.size else 0.0
        return ErrorBoundRecommendation(
            mode="absolute", value=data_range * min(candidates)
        )

    def quadratic_error_bound(self, data) -> ErrorBoundRecommendation:
        """Resolve a quadratic requirement into a conservative bound."""

        import numpy as np

        candidates = tuple(
            node
            for node in self.selected
            if node.kind == "max-pointwise-quadratic-error-bound"
            and isinstance(node.value, (int, float))
            and node.minimum is not None
            and node.maximum is not None
        )
        if not candidates:
            raise KeyError("selected recommendation has no quadratic bound")
        values = np.asarray(data, dtype=np.float64)
        bounds = []
        for node in candidates:
            minimum = float(node.minimum)
            maximum = float(node.maximum)
            if maximum <= minimum:
                bounds.append(0.0)
                continue
            finite = values[np.isfinite(values)]
            if np.any((finite <= minimum) | (finite >= maximum)):
                bounds.append(0.0)
                continue
            scale = 1.0 - (2.0 * (finite - minimum) / (maximum - minimum) - 1.0) ** 2
            bounds.append(float(np.min(scale)) * float(node.value))
        return ErrorBoundRecommendation(mode="absolute", value=max(0.0, min(bounds)))


@dataclass(frozen=True, slots=True)
class ErrorBoundRecommendation:
    """A pointwise error bound that can be applied by :class:`BraceCodec`."""

    mode: ErrorBoundMode
    value: float


def load_recommendations() -> Recommendations:
    """Load the package-provided, strongly typed recommendations."""

    return Recommendations.provide


def plan_recommendation(
    variable: str,
    *,
    level_kind: str = "pressure",
    markers: Mapping[str, None | bool | int | float | str] | None = None,
    recommendations: Recommendations | None = None,
) -> RecommendationPlan:
    """Normalize and select a branch from typed recommendations.

    ``all`` nodes retain every child. ``any`` nodes select one complete branch
    using the explicit strategy preference above; alternatives are never
    flattened into an unintended conjunction.
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
        typed_requirements = source.search(markers=search_markers)
    except KeyError as exc:
        raise KeyError(
            "failed to find a compatible error bound or recommendation",
            search_markers,
        ) from exc
    requirements = tuple(
        _from_config(requirement.get_config())
        for requirement in typed_requirements
    )
    selected_tree = tuple(_select_node(node) for node in requirements)
    return RecommendationPlan(
        variable=variable,
        markers=search_markers,
        recommendations_version=str(source.version),
        requirements=requirements,
        selected_tree=selected_tree,
    )


def recommend_error_bound(
    variable: str,
    *,
    level_kind: str = "pressure",
    markers: Mapping[str, None | bool | int | float | str] | None = None,
    recommendations: Recommendations | None = None,
) -> ErrorBoundRecommendation:
    """Select a pointwise bound from the selected recommendation branch."""

    plan = plan_recommendation(
        variable,
        level_kind=level_kind,
        markers=markers,
        recommendations=recommendations,
    )
    return plan.pointwise_error_bound()


def _from_config(config: Mapping[str, object]) -> RequirementNode:
    children = tuple(
        _from_config(child)
        for child in config.get("requirements", [])  # type: ignore[arg-type]
    )
    return RequirementNode(
        kind=str(config["kind"]),
        children=children,
        value=_number_or_none(config.get("value")),
        minimum=_number_or_none(config.get("minimum")),
        maximum=_number_or_none(config.get("maximum")),
    )


def _number_or_none(value: object) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"requirement value must be numeric; got {value!r}")


def _select_node(node: RequirementNode, data=None) -> RequirementNode:
    if node.kind == "all":
        return RequirementNode(
            kind=node.kind,
            children=tuple(_select_node(child, data) for child in node.children),
        )
    if node.kind == "any":
        if not node.children:
            raise ValueError("cannot select an empty any requirement")
        if data is not None:
            return _select_node(
                max(node.children, key=lambda child: _node_score(child, data)), data
            )
        return _select_node(
            min(node.children, key=lambda child: _node_score(child, data)), data
        )
    return node


def _expand_node(node: RequirementNode, data) -> tuple[RequirementNode, ...]:
    import itertools

    if node.kind == "any":
        choices = []
        for child in node.children:
            choices.extend(_expand_node(child, data))
        return tuple(choices)
    if node.kind == "all":
        children = [tuple(_expand_node(child, data)) for child in node.children]
        return tuple(
            RequirementNode(kind="all", children=tuple(choice))
            for choice in itertools.product(*children)
        )
    return (node,)


def _node_score(node: RequirementNode, data=None):
    leaves = tuple(_leaves((_select_node(node),)))
    if data is not None:
        import numpy as np

        values = np.asarray(data, dtype=np.float64)
        finite = values[np.isfinite(values)]
        mean_abs = float(np.mean(np.abs(finite))) if finite.size else 0.0
        value_range = float(np.ptp(finite)) if finite.size else 0.0
        tolerances = []
        for leaf in leaves:
            value = float(leaf.value) if isinstance(leaf.value, (int, float)) else 0.0
            if leaf.kind in {"max-pointwise-absolute-error-bound", "mean-absolute-error-bound"}:
                tolerances.append(value)
            elif leaf.kind in {"max-pointwise-relative-error-bound", "mean-relative-error-bound"}:
                tolerances.append(value * mean_abs)
            elif leaf.kind in {"max-pointwise-range-relative-error-bound", "mean-range-relative-error-bound"}:
                tolerances.append(value * value_range)
            elif leaf.kind == "max-pointwise-quadratic-error-bound":
                minimum = float(leaf.minimum)
                maximum = float(leaf.maximum)
                if maximum > minimum and finite.size:
                    curve = 1.0 - (2.0 * (finite - minimum) / (maximum - minimum) - 1.0) ** 2
                    tolerances.append(value * max(0.0, float(np.mean(curve))))
                else:
                    tolerances.append(0.0)
            else:
                tolerances.append(0.0)
        preference = max(
            (_KIND_PREFERENCE.get(leaf.kind, 100) for leaf in leaves),
            default=100,
        )
        return (min(tolerances, default=0.0), -preference, -len(leaves), node.kind)
    scores = [_KIND_PREFERENCE.get(leaf.kind, 100) for leaf in leaves]
    return max(scores, default=100), len(scores), node.kind


def _leaves(nodes: Collection[RequirementNode]):
    for node in nodes:
        if node.children:
            yield from _leaves(node.children)
        else:
            yield node
