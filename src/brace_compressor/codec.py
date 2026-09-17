"""The public numcodecs.Codec implementing error-bounded BRACE compression.

Pipeline: predict from already reconstructed neighbors with a
deterministic cold-start value -> quantize residuals (step derived from the
absolute error bound) -> entropy-code symbols (bit-exact, per-block mode
selection) -> verify-and-repair (bounds the decoded error using an
epsilon-clamped effective bound) -> frame container.
"""

from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np
from numcodecs.abc import Codec

from .bound import normalize_missing_value, validate_error_bound, validate_shape
from .container import ContainerError, read_container, write_container
from .mask import apply_mask, extract_mask, pack_mask, unpack_mask
from .model.entropy import decode_symbols, pack_repairs, unpack_repairs
from .model.entropy import encode_symbols as _entropy_encode
from .quant import derive_step
from .verify import verify_and_repair

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
OUTER_COMPRESS_DEFAULT = True  # always-lossless CR-maximizing pass

_MODE_QUANTIZED = 0


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
        field = self._as_field(buf)
        mask = extract_mask(field, self.missing_value, dtype=self.dtype)
        valid_values = field[~mask].astype(self.dtype)
        n_valid = int(valid_values.size)

        bound = self.error_bound
        effective_bound = max(bound, float(np.finfo(self.dtype).eps))
        step = derive_step(bound, dtype=self.dtype)
        prior = np.full(self.shape, 32.0, dtype=self.dtype)
        symbols, recon_rows, origin = self._causal_scan_encode(
            field, mask, prior, step
        )
        # --- Verify-and-repair: compare against the effective bound -------
        grid3d = recon_rows.reshape(self.shape)
        decoded_valid = grid3d[~mask]
        repair_map, _, max_abs_error, _ = verify_and_repair(
            valid_values, decoded_valid, effective_bound, dtype=self.dtype
        )
        repair_positions, repair_values = repair_map.positions, repair_map.values
        # --- Entropy coding (bit-exact; per-block mode selection) ----------
        encoded_syms = _entropy_encode(symbols)
        value_payload = (
            struct.pack("<Q", len(encoded_syms))
            + encoded_syms
            + pack_repairs(repair_positions, repair_values, dtype=self.dtype)
        )
        mode = _MODE_QUANTIZED

        # --- Mask + container -------------------------------------------
        mask_payload = pack_mask(mask)
        header_extra = {
            "shape": list(self.shape),
            "dtype": self.dtype.name,
            "missing_value": "nan" if np.isnan(self.missing_value) else float(self.missing_value),
            "error_bound": float(bound),
            "quant_step": step,
            "origin": float(origin),
            "mode": mode,
            "effective_error_bound": effective_bound,
            "n_valid": int(valid_values.size),
            "n_repaired": int(repair_positions.size),
            "metrics": {
                # FR-012: payload-based metrics (stable, non-self-referential).
                "max_abs_error": max_abs_error,
                "n_repaired": int(repair_positions.size),
                "uncompressed_size": int(field.nbytes),
                "payload_size": int(len(mask_payload) + len(value_payload)),
                "bound_respected": bool(max_abs_error <= effective_bound),
            },
            "codec_version": CODEC_VERSION,
        }
        return write_container(
            header_extra=header_extra,
            mask_payload=mask_payload,
            residual_payload=value_payload,
            model_version=MODEL_VERSION,
            outer_compress=self.outer_compress,
        )

    def decode(self, buf, out=None):
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

        try:
            c = read_container(buf)
        except ContainerError:
            raise
        if c.model_version != MODEL_VERSION:
            raise ValueError(
                f"unsupported scan model version {c.model_version}; "
                f"this codec supports {MODEL_VERSION}"
            )
        h = c.header_extra
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
        origin = float(h["origin"])
        n_valid = int(h["n_valid"])
        missing_value = h["missing_value"]
        sentinel = float("nan") if missing_value == "nan" else float(missing_value)

        mask = unpack_mask(
            c.mask_payload, int(np.prod(self.shape))
        ).reshape(self.shape)

        payload = c.residual_payload
        if len(payload) < 8:
            raise ValueError("residual payload too short")
        (ent_len,) = struct.unpack_from("<Q", payload, 0)
        symbols = decode_symbols(payload[8 : 8 + ent_len], n_valid)
        repair_positions, repair_values, _ = unpack_repairs(
            payload, 8 + ent_len, dtype=self.dtype
        )
        # --- Causal scan (identical walk to encode) ----------------------
        prior = np.full(self.shape, 32.0, dtype=self.dtype)
        recon_rows = self._causal_scan_decode(
            prior,
            mask,
            symbols,
            step,
            origin,
            use_rust=self.dtype in {np.dtype("float32"), np.dtype("float64")},
        )
        decoded_valid = recon_rows.reshape(self.shape)[~mask]
        if repair_positions.size:
            decoded_valid[repair_positions] = repair_values

        field = np.empty(self.shape, dtype=self.dtype)
        if n_valid:
            field[~mask] = decoded_valid
        field = apply_mask(
            decoded=field, mask=mask, missing_value=sentinel, dtype=self.dtype
        )

        if out is not None:
            out[...] = field.reshape(out.shape)
            return out
        return field

    # ------------------------------------------------------------------
    # Causal scan: identical encode/decode walk
    # ------------------------------------------------------------------
    def _causal_scan_encode(self, field, mask, prior, step):
        """Fused encode-side scan (single pass).

        Predicts each valid cell from already-reconstructed
        neighbors (temporal parent, left, top, top-left, top-right),
        quantizes the residual directly on the step-lattice (origin 0;
        entropy is shift-invariant, and the header carries 0.0), and
        materializes the exact decoder state. Uses the Rust extension
        when available (bit-exact), else the pure-Python loops.
        Returns (symbols, recon_rows, origin).
        """
        if _HAS_RUST and self.dtype == np.dtype("float32"):
            symbols, recon_rows = _rs_scan_encode(
                np.ascontiguousarray(field, dtype=np.float32),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(prior, dtype=np.float32),
                float(step),
            )
            return symbols, recon_rows, 0.0
        if _HAS_RUST_F64 and self.dtype == np.dtype("float64"):
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
        inv_step = 1.0 / step_f  # step > 0 in this path
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
                    q = int(math.floor(r * inv_step + 0.5))
                    if q < lo:
                        q = lo
                    elif q > hi:
                        q = hi
                    symbols[k] = q
                    # decoder state: pred + dequantized residual (origin=0)
                    recon_rows[row, xi] = pred + q * step_f
                    k += 1
        return symbols, recon_rows, 0.0

    def _causal_scan_decode(
        self, prior, mask, symbols, step, origin, use_rust=True
    ):
        """Decoder mirror of :meth:`_causal_scan_encode` (same order/math)."""
        if _HAS_RUST and use_rust and self.dtype == np.dtype("float32"):
            return _rs_scan_decode(
                np.ascontiguousarray(prior, dtype=np.float32),
                np.ascontiguousarray(mask),
                np.ascontiguousarray(symbols, dtype=np.int64),
                float(step),
                float(origin),
            )
        if _HAS_RUST_F64 and use_rust and self.dtype == np.dtype("float64"):
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
                    dq = int(symbols[k]) * step_f + origin
                    recon_rows[row, xi] = pred + dq
                    k += 1
        return recon_rows





