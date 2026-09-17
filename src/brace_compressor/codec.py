"""The public numcodecs.Codec implementing error-bounded BRACE compression.

Pipeline: predict from already reconstructed neighbors with a
deterministic cold-start value -> quantize residuals in absolute or logarithmic
relative mode -> entropy-code symbols (bit-exact, per-block mode selection) ->
verify-and-repair -> frame container.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any

import numpy as np
from numcodecs.abc import Codec

from .bound import normalize_missing_value, validate_error_bound, validate_shape
from .container import ContainerError, read_container, write_container
from .constraints import check_requirement, constraint_error_bounds
from .mask import apply_mask, extract_mask, pack_mask, unpack_mask
from .model.entropy import decode_symbols, pack_repairs, unpack_repairs
from .model.entropy import encode_symbols as _entropy_encode
from .quant import derive_step
from .recommendations import (
    ErrorBoundRecommendation,
    plan_recommendation,
    recommend_error_bound,
)
from .verify import (
    measure_errors,
    repair_mean_absolute,
    repair_mean_relative,
    verify_and_repair,
)

# Optional Rust-accelerated scan (bit-exact port of the Python
# loops). Falls back to the pure-Python implementation if the extension
# is not installed.
try:
    from brace_scan import causal_scan_decode as _rs_scan_decode
    from brace_scan import causal_scan_encode as _rs_scan_encode

    _HAS_RUST = True
except Exception:  # pragma: no cover - extension optional
    _HAS_RUST = False

try:
    from brace_scan import causal_scan_decode_f64 as _rs_scan_decode_f64
    from brace_scan import causal_scan_encode_f64 as _rs_scan_encode_f64

    _HAS_RUST_F64 = True
except Exception:  # pragma: no cover - older extension without f64 API
    _HAS_RUST_F64 = False

MODEL_VERSION = 3
CODEC_VERSION = "0.1.0"
RECOMMENDATION_SCHEMA_VERSION = 1
OUTER_COMPRESS_DEFAULT = True  # always-lossless CR-maximizing pass

_MODE_QUANTIZED = 0
_MODE_RELATIVE = 1
_MODE_LOSSLESS = 2
_LOG_OFFSET = 1024.0
_LOCAL_BLOCK_SIZE = 256


def _byte_shuffle(field: np.ndarray) -> bytes:
    """Transpose element bytes so equal-significance bytes are adjacent."""

    raw = np.frombuffer(field.tobytes(order="C"), dtype=np.uint8)
    return raw.reshape(-1, field.dtype.itemsize).T.reshape(-1).tobytes()


def _byte_unshuffle(payload: bytes, dtype: np.dtype, size: int) -> np.ndarray:
    """Invert :func:`_byte_shuffle` into a native typed array."""

    expected = size * dtype.itemsize
    if len(payload) != expected:
        raise ValueError("invalid lossless transformed payload length")
    shuffled = np.frombuffer(payload, dtype=np.uint8).reshape(dtype.itemsize, size)
    return np.ascontiguousarray(shuffled.T.reshape(-1)).view(dtype)


class BraceCodec(Codec):
    """Error-bounded reconstructed-neighbor codec for gridded fields.

    Both 2-D ``(lat, lon)`` slices and 3-D ``(time, lat, lon)``
    fields are accepted. Two-dimensional inputs are encoded as one time slice.

    Implements the numcodecs ``Codec`` contract: ``codec_id``, ``encode``,
    ``decode(buf, out=None)``, ``get_config``, ``from_config``.
    """

    codec_id = "brace"

    def __init__(
        self,
        shape,
        error_bound,
        missing_value="nan",
        dtype="float32",
        outer_compress: bool = OUTER_COMPRESS_DEFAULT,
        error_bound_mode: str = "absolute",
    ):
        self.shape = validate_shape(shape)
        self.error_bound = validate_error_bound(error_bound)
        self.missing_value = normalize_missing_value(missing_value)
        try:
            self.dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}") from exc
        if self.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}")
        self.outer_compress = bool(outer_compress)
        if error_bound_mode not in {"absolute", "relative"}:
            raise ValueError("error_bound_mode must be 'absolute' or 'relative'")
        self.error_bound_mode = error_bound_mode
        self.last_timings: dict[str, float] = {}
        self.recommendation_plan = None

    @classmethod
    def from_recommendation(
        cls,
        *,
        shape,
        variable: str,
        level_kind: str = "pressure",
        markers=None,
        recommendations=None,
        **kwargs,
    ) -> "BraceCodec":
        """Create a codec using the typed recommendation for ``variable``."""

        plan = plan_recommendation(
            variable,
            level_kind=level_kind,
            markers=markers,
            recommendations=recommendations,
        )
        try:
            recommendation = plan.pointwise_error_bound()
        except KeyError:
            recommendation = ErrorBoundRecommendation(mode="absolute", value=0.0)
            quadratic_values = tuple(
                float(node.value)
                for node in plan.selected
                if node.kind == "max-pointwise-quadratic-error-bound"
                and isinstance(node.value, (int, float))
            )
            if quadratic_values:
                recommendation = ErrorBoundRecommendation(
                    mode="absolute", value=max(quadratic_values)
                )
            if not any(
                node.kind
                in {
                    "max-pointwise-range-relative-error-bound",
                    "mean-range-relative-error-bound",
                    "max-pointwise-quadratic-error-bound",
                    "data-limits",
                    "isovalue",
                    "missing-value",
                    "lossless",
                }
                for node in plan.selected
            ):
                raise
        missing_values = tuple(
            node.value
            for node in plan.selected
            if node.kind == "missing-value" and node.value is not None
        )
        if missing_values:
            recommended_missing = float(missing_values[0])
            if "missing_value" not in kwargs:
                kwargs["missing_value"] = recommended_missing
                configured_missing = recommended_missing
            else:
                configured_missing = kwargs["missing_value"]
            configured_missing = normalize_missing_value(configured_missing)
            same_missing = (
                np.isnan(recommended_missing) and np.isnan(configured_missing)
            ) or recommended_missing == configured_missing
            if not same_missing:
                raise ValueError(
                    "missing-value recommendation conflicts with codec missing_value"
                )
            kwargs["missing_value"] = configured_missing
        explicit_base_bound = "error_bound" in kwargs
        base_bound = kwargs.pop("error_bound", recommendation.value)
        if not explicit_base_bound and any(
            node.kind == "mean-absolute-error-bound"
            for node in plan.selected
        ):
            base_bound = recommendation.value * math.sqrt(max(1, int(np.prod(shape))))
        if not explicit_base_bound and any(
            node.kind == "mean-relative-error-bound" for node in plan.selected
        ):
            base_bound = recommendation.value * math.sqrt(max(1, int(np.prod(shape))))
        explicit_base_mode = "error_bound_mode" in kwargs
        base_mode = kwargs.pop("error_bound_mode", recommendation.mode)
        codec = cls(
            shape=shape,
            error_bound=base_bound,
            error_bound_mode=base_mode,
            **kwargs,
        )
        codec.recommendation_plan = plan
        codec._recommendation_explicit_bound = explicit_base_bound
        codec._recommendation_explicit_mode = explicit_base_mode
        return codec

    # ------------------------------------------------------------------
    # Configuration contract (numcodecs.Codec)
    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """JSON-serializable configuration including the 'id' field."""
        missing: Any = (
            "nan" if np.isnan(self.missing_value) else float(self.missing_value)
        )
        return {
            "id": self.codec_id,
            "shape": list(self.shape),
            "error_bound": float(self.error_bound),
            "missing_value": missing,
            "dtype": self.dtype.name,
            "outer_compress": self.outer_compress,
            "error_bound_mode": self.error_bound_mode,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BraceCodec":
        cfg = dict(config)
        codec_id = cfg.pop("id", None)
        if codec_id is not None and codec_id != cls.codec_id:
            raise ValueError(
                f"config id mismatch: expected {cls.codec_id!r}, got {codec_id!r}"
            )
        try:
            shape = cfg.pop("shape")
            error_bound = cfg.pop("error_bound")
        except KeyError as exc:
            raise ValueError(f"missing required config key: {exc.args[0]!r}") from exc
        return cls(shape=shape, error_bound=error_bound, **cfg)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------
    def _as_field(self, buf) -> np.ndarray:
        arr = np.asarray(buf)
        expected = int(np.prod(self.shape))
        if arr.size != expected:
            raise ValueError(
                f"encode buffer has {arr.size} elements; expected {expected} "
                f"(shape {self.shape})"
            )
        if arr.dtype != self.dtype:
            if arr.dtype.kind == "f":
                arr = arr.astype(self.dtype)
            elif arr.nbytes == expected * self.dtype.itemsize:
                arr = arr.view(self.dtype)
            else:
                raise ValueError(
                    f"cannot interpret buffer of dtype {arr.dtype} as {self.dtype.name} field"
                )
        field = arr.reshape(self.shape)
        if not field.flags["C_CONTIGUOUS"]:
            field = np.ascontiguousarray(field)
        return np.ascontiguousarray(field, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, buf) -> bytes:
        encode_start = time.perf_counter()
        field = self._as_field(buf)
        if (
            self.recommendation_plan is not None
            and not getattr(self, "_candidate_encoding", False)
            and any(node.kind == "any" for node in self.recommendation_plan.requirements)
        ):
            candidates = self.recommendation_plan.candidate_plans_for_data(field)
            valid_streams = []
            for candidate_plan in candidates:
                candidate = type(self)(
                    shape=self.shape,
                    error_bound=self.error_bound,
                    missing_value=self.missing_value,
                    dtype=self.dtype,
                    outer_compress=self.outer_compress,
                    error_bound_mode=self.error_bound_mode,
                )
                candidate.recommendation_plan = candidate_plan
                candidate._candidate_encoding = True
                candidate._recommendation_explicit_bound = getattr(
                    self, "_recommendation_explicit_bound", False
                )
                candidate._recommendation_explicit_mode = getattr(
                    self, "_recommendation_explicit_mode", False
                )
                if not candidate._recommendation_explicit_bound:
                    try:
                        bound = candidate_plan.pointwise_error_bound()
                        candidate.error_bound = bound.value
                        if not candidate._recommendation_explicit_mode:
                            candidate.error_bound_mode = bound.mode
                    except KeyError:
                        pass
                stream = candidate.encode(field)
                checks = read_container(stream).header_extra.get(
                    "recommendation_checks", []
                )
                if all(bool(check.get("passed", False)) for check in checks):
                    valid_streams.append((len(stream), candidate_plan, stream))
            if not valid_streams:
                raise ValueError("no recommendation branch produced a valid stream")
            _, self.recommendation_plan, encoded = min(
                valid_streams, key=lambda item: item[0]
            )
            if not getattr(self, "_recommendation_explicit_bound", False):
                try:
                    selected_bound = self.recommendation_plan.pointwise_error_bound()
                    self.error_bound = selected_bound.value
                    if not getattr(self, "_recommendation_explicit_mode", False):
                        self.error_bound_mode = selected_bound.mode
                except KeyError:
                    pass
            return encoded
        if self.recommendation_plan is not None and any(
            node.kind == "any" for node in self.recommendation_plan.requirements
        ):
            self.recommendation_plan = self.recommendation_plan.select_for_data(field)
            if not getattr(self, "_recommendation_explicit_bound", False):
                try:
                    selected_bound = self.recommendation_plan.pointwise_error_bound()
                    self.error_bound = selected_bound.value
                    if not getattr(self, "_recommendation_explicit_mode", False):
                        self.error_bound_mode = selected_bound.mode
                except KeyError:
                    pass
        if self.recommendation_plan is not None and any(
            node.kind == "lossless" for node in self.recommendation_plan.selected
        ):
            header_extra = {
                "shape": list(self.shape),
                "dtype": self.dtype.name,
                "missing_value": "nan"
                if np.isnan(self.missing_value)
                else float(self.missing_value),
                "error_bound": 0.0,
                "error_bound_mode": "absolute",
                "quant_step": 0.0,
                "origin": 0.0,
                "mode": _MODE_LOSSLESS,
                "strategy": "lossless-byte-shuffle",
                "lossless_transform": "byte-shuffle",
                "effective_error_bound": 0.0,
                "n_valid": int(field.size),
                "n_symbols": 0,
                "n_repaired": 0,
                "mask_parts": {"missing": 0},
                "metrics": {
                    "max_abs_error": 0.0,
                    "max_relative_error": 0.0,
                    "n_repaired": 0,
                    "uncompressed_size": int(field.nbytes),
                    "payload_size": int(field.nbytes),
                    "bound_respected": True,
                },
                "codec_version": CODEC_VERSION,
                "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
            }
            if self.recommendation_plan is not None:
                header_extra["recommendation_plan"] = self.recommendation_plan.get_config()
                header_extra["recommendation_checks"] = [
                    check_requirement(node, field, field).get_config()
                    for node in self.recommendation_plan.selected_tree
                ]
            encoded = write_container(
                header_extra=header_extra,
                mask_payload=b"",
                residual_payload=_byte_shuffle(field),
                model_version=MODEL_VERSION,
                outer_compress=self.outer_compress,
            )
            self.last_timings = {
                "encode_total_s": time.perf_counter() - encode_start,
            }
            return encoded
        stage_start = time.perf_counter()
        mask = extract_mask(field, self.missing_value, dtype=self.dtype)
        valid_values = field[~mask].astype(self.dtype)
        n_valid = int(valid_values.size)
        mask_time = time.perf_counter() - stage_start

        bound = self.error_bound
        if self.recommendation_plan is not None and any(
            node.kind
            in {
                "max-pointwise-range-relative-error-bound",
                "mean-range-relative-error-bound",
            }
            for node in self.recommendation_plan.selected
        ):
            bound = self.recommendation_plan.range_relative_error_bound(field).value
            self.error_bound = bound
        relative = self.error_bound_mode == "relative"
        effective_bound = max(bound, float(np.finfo(self.dtype).eps))
        zero_mask = np.zeros(self.shape, dtype=bool)
        negative_mask = np.zeros(self.shape, dtype=bool)
        scan_field = field
        scan_mask = mask
        if relative:
            zero_mask = (~mask) & (field == 0.0)
            negative_mask = (~mask) & (field < 0.0)
            scan_mask = mask | zero_mask
            scan_field = np.zeros_like(field)
            nonzero = ~scan_mask
            scan_field[nonzero] = (
                np.log(np.abs(field[nonzero])) + _LOG_OFFSET
            ).astype(self.dtype)
            step = 2.0 * math.log1p(effective_bound)
            prior = np.full(self.shape, _LOG_OFFSET, dtype=self.dtype)
        else:
            step = derive_step(bound, dtype=self.dtype)
            prior = np.full(self.shape, 32.0, dtype=self.dtype)
        block_steps = None
        if self.recommendation_plan is not None and any(
            node.kind == "max-pointwise-quadratic-error-bound"
            for node in self.recommendation_plan.selected
        ):
            policy_bounds = [
                constraint_error_bounds(node, field)
                for node in self.recommendation_plan.selected_tree
            ]
            if policy_bounds:
                local_bounds = np.minimum.reduce(policy_bounds).reshape(-1)
                n_blocks = (field.size + _LOCAL_BLOCK_SIZE - 1) // _LOCAL_BLOCK_SIZE
                block_steps = []
                for block_index in range(n_blocks):
                    start = block_index * _LOCAL_BLOCK_SIZE
                    stop = min(start + _LOCAL_BLOCK_SIZE, field.size)
                    values = local_bounds[start:stop]
                    finite = values[np.isfinite(values) & (values > 0.0)]
                    local_bound = float(finite.min()) if finite.size else float(
                        np.finfo(self.dtype).eps
                    )
                    block_steps.append(derive_step(local_bound, dtype=self.dtype))
        stage_start = time.perf_counter()
        symbols, recon_rows, origin = self._causal_scan_encode(
            scan_field,
            scan_mask,
            prior,
            step,
            block_steps=block_steps,
            use_rust=not relative,
        )
        scan_time = time.perf_counter() - stage_start
        # --- Verify-and-repair: compare against the effective bound -------
        grid3d = recon_rows.reshape(self.shape)
        if relative:
            decoded_grid = np.zeros(self.shape, dtype=self.dtype)
            decoded_grid[~scan_mask] = np.exp(
                grid3d[~scan_mask].astype(np.float64) - _LOG_OFFSET
            ).astype(self.dtype)
            decoded_grid[negative_mask] *= -1.0
            decoded_valid = decoded_grid[~mask]
            effective_bound_values = (
                np.abs(valid_values.astype(np.float64)) * effective_bound
            )
        else:
            decoded_valid = grid3d[~mask]
            effective_bound_values = effective_bound
        if self.recommendation_plan is not None:
            recommendation_bounds = [
                constraint_error_bounds(node, field)
                for node in self.recommendation_plan.selected_tree
            ]
            if recommendation_bounds:
                policy_bound = np.minimum.reduce(recommendation_bounds)
                if np.isscalar(effective_bound_values):
                    effective_bound_values = np.minimum(
                        policy_bound[~mask], float(effective_bound_values)
                    )
                else:
                    effective_bound_values = np.minimum(
                        policy_bound[~mask], effective_bound_values
                    )
        stage_start = time.perf_counter()
        repair_map, verified_errors, max_abs_error, _ = verify_and_repair(
            valid_values, decoded_valid, effective_bound_values, dtype=self.dtype
        )
        mean_absolute_values = ()
        if self.recommendation_plan is not None:
            mean_absolute_values = tuple(
                float(node.value)
                for node in self.recommendation_plan.selected
                if node.kind == "mean-absolute-error-bound"
                and isinstance(node.value, (int, float))
            )
        if mean_absolute_values:
            candidate = decoded_valid.copy()
            if repair_map.positions.size:
                candidate[repair_map.positions] = repair_map.values
            aggregate_repairs, verified_errors = repair_mean_absolute(
                valid_values,
                candidate,
                min(mean_absolute_values),
                dtype=self.dtype,
            )
            if aggregate_repairs.positions.size:
                repair_positions = np.concatenate(
                    [repair_map.positions, aggregate_repairs.positions]
                )
                repair_values = np.concatenate(
                    [repair_map.values, aggregate_repairs.values]
                )
                repair_map.positions = repair_positions
                repair_map.values = repair_values
            max_abs_error = float(verified_errors.max()) if verified_errors.size else 0.0
        mean_relative_values = ()
        if self.recommendation_plan is not None:
            mean_relative_values = tuple(
                float(node.value)
                for node in self.recommendation_plan.selected
                if node.kind == "mean-relative-error-bound"
                and isinstance(node.value, (int, float))
            )
        if mean_relative_values:
            candidate = decoded_valid.copy()
            if repair_map.positions.size:
                candidate[repair_map.positions] = repair_map.values
            aggregate_repairs, verified_errors = repair_mean_relative(
                valid_values,
                candidate,
                min(mean_relative_values),
                dtype=self.dtype,
            )
            if aggregate_repairs.positions.size:
                repair_positions = np.concatenate(
                    [repair_map.positions, aggregate_repairs.positions]
                )
                repair_values = np.concatenate(
                    [repair_map.values, aggregate_repairs.values]
                )
                repair_map.positions = repair_positions
                repair_map.values = repair_values
            max_abs_error = float(verified_errors.max()) if verified_errors.size else 0.0
        verify_time = time.perf_counter() - stage_start
        repair_positions, repair_values = repair_map.positions, repair_map.values
        recommendation_checks = None
        if self.recommendation_plan is not None:
            repaired_valid = decoded_valid.copy()
            if repair_positions.size:
                repaired_valid[repair_positions] = repair_values
            reconstructed = np.empty_like(field)
            reconstructed[~mask] = repaired_valid
            reconstructed[mask] = self.missing_value
            recommendation_checks = [
                check_requirement(node, field, reconstructed).get_config()
                for node in self.recommendation_plan.selected_tree
            ]
        # --- Entropy coding (bit-exact; per-block mode selection) ----------
        stage_start = time.perf_counter()
        encoded_syms = _entropy_encode(symbols)
        value_payload = (
            struct.pack("<Q", len(encoded_syms))
            + encoded_syms
            + pack_repairs(repair_positions, repair_values, dtype=self.dtype)
        )
        entropy_time = time.perf_counter() - stage_start
        mode = _MODE_RELATIVE if relative else _MODE_QUANTIZED

        # --- Mask + container -------------------------------------------
        mask_payload = pack_mask(mask)
        mask_parts = {"missing": len(mask_payload)}
        if relative:
            zero_payload = pack_mask(zero_mask)
            negative_payload = pack_mask(negative_mask)
            mask_payload += zero_payload + negative_payload
            mask_parts.update(zero=len(zero_payload), negative=len(negative_payload))
        header_extra = {
            "shape": list(self.shape),
            "dtype": self.dtype.name,
            "missing_value": "nan" if np.isnan(self.missing_value) else float(self.missing_value),
            "error_bound": float(bound),
            "error_bound_mode": self.error_bound_mode,
            "quant_step": step,
            "block_size": _LOCAL_BLOCK_SIZE if block_steps is not None else None,
            "block_steps": block_steps,
            "origin": float(origin),
            "mode": mode,
            "strategy": "relative-scalar" if relative else (
                "absolute-block-local" if block_steps is not None else "absolute-scalar"
            ),
            "effective_error_bound": float(effective_bound),
            "n_valid": int(valid_values.size),
            "n_symbols": int(symbols.size),
            "n_repaired": int(repair_positions.size),
            "mask_parts": mask_parts,
            "metrics": {
                # FR-012: payload-based metrics (stable, non-self-referential).
                "max_abs_error": max_abs_error,
                "max_relative_error": float(
                    np.max(
                        np.divide(
                            verified_errors.astype(np.float64),
                            np.abs(valid_values.astype(np.float64)),
                            out=np.zeros_like(verified_errors, dtype=np.float64),
                            where=valid_values != 0,
                        )
                    )
                ) if relative and verified_errors.size else 0.0,
                "n_repaired": int(repair_positions.size),
                "uncompressed_size": int(field.nbytes),
                "payload_size": int(len(mask_payload) + len(value_payload)),
                "bound_respected": bool(
                    np.all(verified_errors <= effective_bound_values)
                ),
            },
            "codec_version": CODEC_VERSION,
            "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
        }
        if self.recommendation_plan is not None:
            header_extra["recommendation_plan"] = self.recommendation_plan.get_config()
            header_extra["recommendation_checks"] = recommendation_checks
        stage_start = time.perf_counter()
        encoded = write_container(
            header_extra=header_extra,
            mask_payload=mask_payload,
            residual_payload=value_payload,
            model_version=MODEL_VERSION,
            outer_compress=self.outer_compress,
        )
        self.last_timings = {
            "encode_mask_s": mask_time,
            "encode_scan_s": scan_time,
            "encode_verify_s": verify_time,
            "encode_entropy_s": entropy_time,
            "encode_container_s": time.perf_counter() - stage_start,
            "encode_total_s": time.perf_counter() - encode_start,
        }
        return encoded

    def decode(self, buf, out=None):
        decode_start = time.perf_counter()
        if out is not None:
            out = np.asarray(out)
            expected = int(np.prod(self.shape)) * self.dtype.itemsize
            if out.nbytes != expected:
                raise ValueError(
                    f"out buffer must be exactly {expected} bytes "
                    f"(shape {self.shape}, {self.dtype.name}); got {out.nbytes}"
                )
            if not out.flags["C_CONTIGUOUS"] or out.dtype != self.dtype:
                raise ValueError(f"out buffer must be C-contiguous {self.dtype.name}")

        stage_start = time.perf_counter()
        try:
            c = read_container(buf)
        except ContainerError:
            raise
        container_time = time.perf_counter() - stage_start
        if c.model_version != MODEL_VERSION:
            raise ValueError(
                f"unsupported scan model version {c.model_version}; "
                f"this codec supports {MODEL_VERSION}"
            )
        h = c.header_extra
        if not isinstance(h, dict):
            raise ValueError("container header-extra must be an object")
        schema_version = h.get("recommendation_schema_version", 1)
        if schema_version != RECOMMENDATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported recommendation schema version {schema_version!r}"
            )
        stream_dtype = np.dtype(h.get("dtype", "float32"))
        if stream_dtype != self.dtype:
            raise ValueError(
                f"container dtype {stream_dtype.name} does not match codec dtype "
                f"{self.dtype.name}"
            )
        shape = tuple(h["shape"])
        if shape != self.shape:
            raise ValueError(
                f"container shape {shape} does not match codec shape {self.shape}"
            )
        step = float(h["quant_step"])
        block_size = h.get("block_size")
        block_steps = h.get("block_steps")
        if block_steps is not None:
            if block_size != _LOCAL_BLOCK_SIZE or len(block_steps) != (
                int(np.prod(self.shape)) + _LOCAL_BLOCK_SIZE - 1
            ) // _LOCAL_BLOCK_SIZE:
                raise ValueError("invalid local quantization metadata")
            block_steps = [float(value) for value in block_steps]
            if not block_steps or not all(
                math.isfinite(value) and value > 0.0 for value in block_steps
            ):
                raise ValueError("local quantization steps must be finite and positive")
        origin = float(h["origin"])
        n_valid = int(h["n_valid"])
        if n_valid < 0 or n_valid > int(np.prod(self.shape)):
            raise ValueError("invalid valid-element count")
        mode = int(h.get("mode", _MODE_QUANTIZED))
        if mode not in {_MODE_QUANTIZED, _MODE_RELATIVE, _MODE_LOSSLESS}:
            raise ValueError(f"unsupported codec mode {mode}")
        strategy = h.get("strategy", "absolute-scalar")
        valid_strategies = {
            _MODE_LOSSLESS: {"lossless-raw", "lossless-byte-shuffle"},
            _MODE_RELATIVE: {"relative-scalar"},
            _MODE_QUANTIZED: {"absolute-scalar", "absolute-block-local"},
        }
        if strategy not in valid_strategies[mode]:
            raise ValueError(f"strategy {strategy!r} is incompatible with mode {mode}")
        stream_mode = h.get(
            "error_bound_mode",
            "relative" if mode == _MODE_RELATIVE else "absolute",
        )
        if stream_mode != self.error_bound_mode:
            raise ValueError(
                f"container error_bound_mode {stream_mode!r} does not match codec "
                f"mode {self.error_bound_mode!r}"
            )
        if mode == _MODE_LOSSLESS:
            expected_bytes = int(np.prod(self.shape)) * self.dtype.itemsize
            if len(c.mask_payload) != 0 or len(c.residual_payload) != expected_bytes:
                raise ValueError("invalid lossless payload length")
            transform = h.get("lossless_transform", "raw")
            if transform == "byte-shuffle":
                field = _byte_unshuffle(
                    c.residual_payload, self.dtype, int(np.prod(self.shape))
                ).reshape(self.shape)
            elif transform == "raw":
                field = np.frombuffer(c.residual_payload, dtype=self.dtype).copy().reshape(
                    self.shape
                )
            else:
                raise ValueError(f"unsupported lossless transform {transform!r}")
            self.last_timings = {
                "decode_container_s": container_time,
                "decode_total_s": time.perf_counter() - decode_start,
            }
            if out is not None:
                out[...] = field.reshape(out.shape)
                return out
            return field
        n_symbols = int(h.get("n_symbols", n_valid))
        missing_value = h["missing_value"]
        sentinel = float("nan") if missing_value == "nan" else float(missing_value)

        stage_start = time.perf_counter()
        n_elements = int(np.prod(self.shape))
        if mode == _MODE_RELATIVE:
            mask_parts = h.get("mask_parts")
            if not mask_parts:
                raise ValueError("relative stream is missing mask metadata")
            offset = 0
            missing_size = int(mask_parts["missing"])
            zero_size = int(mask_parts["zero"])
            negative_size = int(mask_parts["negative"])
            mask = unpack_mask(c.mask_payload[offset : offset + missing_size], n_elements).reshape(self.shape)
            offset += missing_size
            zero_mask = unpack_mask(c.mask_payload[offset : offset + zero_size], n_elements).reshape(self.shape)
            offset += zero_size
            negative_mask = unpack_mask(c.mask_payload[offset : offset + negative_size], n_elements).reshape(self.shape)
            offset += negative_size
            if offset != len(c.mask_payload):
                raise ValueError("relative mask payload has trailing bytes")
            if np.any(zero_mask & (mask | negative_mask)) or np.any(negative_mask & mask):
                raise ValueError("relative special-value masks overlap")
            scan_mask = mask | zero_mask
        else:
            mask = unpack_mask(c.mask_payload, n_elements).reshape(self.shape)
            scan_mask = mask
        mask_time = time.perf_counter() - stage_start

        payload = c.residual_payload
        if len(payload) < 8:
            raise ValueError("residual payload too short")
        (ent_len,) = struct.unpack_from("<Q", payload, 0)
        if ent_len > len(payload) - 8:
            raise ValueError("entropy section exceeds residual payload")
        if n_symbols > n_valid:
            raise ValueError("symbol count exceeds valid-element count")
        stage_start = time.perf_counter()
        symbols = decode_symbols(payload[8 : 8 + ent_len], n_symbols)
        repair_positions, repair_values, _ = unpack_repairs(
            payload, 8 + ent_len, dtype=self.dtype
        )
        repair_end = 8 + ent_len + 4 + repair_positions.size * (8 + self.dtype.itemsize)
        if repair_end != len(payload):
            raise ValueError("repair payload has trailing bytes")
        if np.any(repair_positions < 0) or np.any(repair_positions >= n_valid):
            raise ValueError("repair index outside valid-element range")
        entropy_time = time.perf_counter() - stage_start
        # --- Causal scan (identical walk to encode) ----------------------
        relative = mode == _MODE_RELATIVE
        prior = np.full(
            self.shape, _LOG_OFFSET if relative else 32.0, dtype=self.dtype
        )
        stage_start = time.perf_counter()
        recon_rows = self._causal_scan_decode(
            prior,
            scan_mask,
            symbols,
            step,
            origin,
            block_steps=block_steps,
            use_rust=not relative and self.dtype in {np.dtype("float32"), np.dtype("float64")},
        )
        scan_time = time.perf_counter() - stage_start
        if relative:
            field = np.zeros(self.shape, dtype=self.dtype)
            field[~scan_mask] = np.exp(
                recon_rows.reshape(self.shape)[~scan_mask].astype(np.float64)
                - _LOG_OFFSET
            ).astype(self.dtype)
            field[negative_mask] *= -1.0
            decoded_valid = field[~mask]
        else:
            decoded_valid = recon_rows.reshape(self.shape)[~mask]
        if repair_positions.size:
            decoded_valid[repair_positions] = repair_values

        if not relative:
            field = np.empty(self.shape, dtype=self.dtype)
            if n_valid:
                field[~mask] = decoded_valid
        stage_start = time.perf_counter()
        field = apply_mask(
            decoded=field, mask=mask, missing_value=sentinel, dtype=self.dtype
        )
        restore_time = time.perf_counter() - stage_start
        self.last_timings = {
            "decode_container_s": container_time,
            "decode_mask_s": mask_time,
            "decode_entropy_s": entropy_time,
            "decode_scan_s": scan_time,
            "decode_restore_s": restore_time,
            "decode_total_s": time.perf_counter() - decode_start,
        }

        if out is not None:
            out[...] = field.reshape(out.shape)
            return out
        return field

    # ------------------------------------------------------------------
    # Causal scan: identical encode/decode walk
    # ------------------------------------------------------------------
    def _causal_scan_encode(
        self, field, mask, prior, step, block_steps=None, use_rust=True
    ):
        """Fused encode-side scan (single pass).

        Predicts each valid cell from already-reconstructed
        neighbors (temporal parent, left, top, top-left, top-right),
        quantizes the residual directly on the step-lattice (origin 0;
        entropy is shift-invariant, and the header carries 0.0), and
        materializes the exact decoder state. Uses the Rust extension
        when available (bit-exact), else the pure-Python loops.
        Returns (symbols, recon_rows, origin).
        """
        if block_steps is None and use_rust and _HAS_RUST and self.dtype == np.dtype("float32"):
            symbols, recon_rows = _rs_scan_encode(
                np.ascontiguousarray(field, dtype=np.float32),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(prior, dtype=np.float32),
                float(step),
            )
            return symbols, recon_rows, 0.0
        if block_steps is None and use_rust and _HAS_RUST_F64 and self.dtype == np.dtype("float64"):
            symbols, recon_rows = _rs_scan_encode_f64(
                np.ascontiguousarray(field, dtype=np.float64),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(prior, dtype=np.float64),
                float(step),
            )
            return symbols, recon_rows, 0.0
        t, lat, lon = self.shape
        n_valid = int(mask.size - int(mask.sum()))
        symbols = np.zeros(n_valid, dtype=np.int64)
        recon_rows = np.zeros((t * lat, lon), dtype=np.float64)
        fld = field
        msk = mask
        step_f = float(step)
        lo = -(1 << 31) + 1
        hi = (1 << 31) - 1
        prior_np = np.asarray(prior, dtype=np.float64)
        k = 0
        for ti in range(t):
            base = ti * lat
            for yi in range(lat):
                row = base + yi
                has_top = yi > 0
                top_row = row - 1
                has_time = ti > 0
                time_row = row - lat
                for xi in range(lon):
                    if msk[ti, yi, xi]:
                        continue
                    # neighbors: all already reconstructed (decode
                    # holds the same state at this point of the scan).
                    # Longitude-local continuity dominates this traversal.
                    preds = []
                    wts = []
                    if xi > 0 and recon_rows[row, xi - 1] != 0.0:
                        preds.append(recon_rows[row, xi - 1]); wts.append(8.0)
                    if has_top and recon_rows[top_row, xi] != 0.0:
                        preds.append(recon_rows[top_row, xi]); wts.append(2.0)
                    if has_top and xi > 0 and recon_rows[top_row, xi - 1] != 0.0:
                        preds.append(recon_rows[top_row, xi - 1]); wts.append(1.0)
                    if has_top and xi < lon - 1 and recon_rows[top_row, xi + 1] != 0.0:
                        preds.append(recon_rows[top_row, xi + 1]); wts.append(1.0)
                    if has_time and recon_rows[time_row, xi] != 0.0:
                        preds.append(recon_rows[time_row, xi]); wts.append(1.0)
                    if preds:
                        pred = sum(p * w for p, w in zip(preds, wts)) / sum(wts)
                    else:
                        pred = prior_np[ti, yi, xi]
                    r = float(fld[ti, yi, xi]) - pred
                    flat_index = (ti * lat + yi) * lon + xi
                    local_step = (
                        float(block_steps[flat_index // _LOCAL_BLOCK_SIZE])
                        if block_steps is not None
                        else step_f
                    )
                    q = int(math.floor(r / local_step + 0.5))
                    if q < lo:
                        q = lo
                    elif q > hi:
                        q = hi
                    symbols[k] = q
                    # decoder state: pred + dequantized residual (origin=0)
                    recon_rows[row, xi] = pred + q * local_step
                    k += 1
        return symbols, recon_rows, 0.0

    def _causal_scan_decode(
        self, prior, mask, symbols, step, origin, block_steps=None, use_rust=True
    ):
        """Decoder mirror of :meth:`_causal_scan_encode` (same order/math)."""
        if block_steps is None and _HAS_RUST and use_rust and self.dtype == np.dtype("float32"):
            return _rs_scan_decode(
                np.ascontiguousarray(prior, dtype=np.float32),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(symbols, dtype=np.int64),
                float(step),
                float(origin),
            )
        if block_steps is None and _HAS_RUST_F64 and use_rust and self.dtype == np.dtype("float64"):
            return _rs_scan_decode_f64(
                np.ascontiguousarray(prior, dtype=np.float64),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(symbols, dtype=np.int64),
                float(step),
                float(origin),
            )
        t, lat, lon = self.shape
        recon_rows = np.zeros((t * lat, lon), dtype=np.float64)
        msk = mask
        step_f = float(step)
        prior_np = np.asarray(prior, dtype=np.float64)
        k = 0
        for ti in range(t):
            base = ti * lat
            for yi in range(lat):
                row = base + yi
                has_top = yi > 0
                top_row = row - 1
                has_time = ti > 0
                time_row = row - lat
                for xi in range(lon):
                    if msk[ti, yi, xi]:
                        continue
                    preds = []
                    wts = []
                    if xi > 0 and recon_rows[row, xi - 1] != 0.0:
                        preds.append(recon_rows[row, xi - 1]); wts.append(8.0)
                    if has_top and recon_rows[top_row, xi] != 0.0:
                        preds.append(recon_rows[top_row, xi]); wts.append(2.0)
                    if has_top and xi > 0 and recon_rows[top_row, xi - 1] != 0.0:
                        preds.append(recon_rows[top_row, xi - 1]); wts.append(1.0)
                    if has_top and xi < lon - 1 and recon_rows[top_row, xi + 1] != 0.0:
                        preds.append(recon_rows[top_row, xi + 1]); wts.append(1.0)
                    if has_time and recon_rows[time_row, xi] != 0.0:
                        preds.append(recon_rows[time_row, xi]); wts.append(1.0)
                    if preds:
                        pred = sum(p * w for p, w in zip(preds, wts)) / sum(wts)
                    else:
                        pred = prior_np[ti, yi, xi]
                    flat_index = (ti * lat + yi) * lon + xi
                    local_step = (
                        float(block_steps[flat_index // _LOCAL_BLOCK_SIZE])
                        if block_steps is not None
                        else step_f
                    )
                    dq = int(symbols[k]) * local_step + origin
                    recon_rows[row, xi] = pred + dq
                    k += 1
        return recon_rows





