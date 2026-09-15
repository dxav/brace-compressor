"""The public numcodecs.Codec implementing error-bounded HOAPS wvpa compression.

Pipeline (research.md R4): predict (space-time transformer, mask-driven and
identical at encode/decode) -> quantize residuals (step derived from the
absolute error bound) -> entropy-code symbols (bit-exact, per-block mode
selection) -> verify-and-repair (bounds the decoded error; skipped when
bound == 0 per the FR-003 exemption) -> frame container.
"""

from __future__ import annotations

import struct

import numpy as np

from .bound import normalize_missing_value, validate_error_bound, validate_shape
from .container import ContainerError, read_container, write_container
from .mask import apply_mask, extract_mask, pack_mask, unpack_mask
from .model.entropy import decode_symbols, pack_repairs, unpack_repairs
from .model.entropy import encode_symbols as _entropy_encode
from .model.transformer import MODEL_VERSION as TRANSFORMER_MODEL_VERSION
from .model.transformer import TransformerPredictor
from .quant import dequantize, derive_step, quantize
from .verify import verify_and_repair

MODEL_VERSION = TRANSFORMER_MODEL_VERSION
CODEC_VERSION = "0.1.0"
OUTER_COMPRESS_DEFAULT = True  # always-lossless CR-maximizing pass (FR-018)

_MODE_QUANTIZED = 0
_MODE_EXACT = 1  # bound == 0: raw float32 bit patterns (tightest representation)


class HoapsWvpaCodec:
    """Error-bounded transformer-based codec for HOAPS wvpa gridded fields.

    Implements the numcodecs ``Codec`` contract: ``codec_id``, ``encode``,
    ``decode(buf, out=None)``, ``get_config``, ``from_config``.
    """

    codec_id = "hoaps-wvpa"

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
        if dtype != "float32":
            raise ValueError(f"dtype must be 'float32' in v1; got {dtype!r}")
        self.dtype = np.dtype("float32")
        self.outer_compress = bool(outer_compress)
        # Space-time transformer predictor: consumes ONLY the missingness
        # mask, so encode and decode compute identical predictions.
        self._predictor = TransformerPredictor()

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
            "dtype": "float32",
            "outer_compress": self.outer_compress,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HoapsWvpaCodec":
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
        if arr.dtype != np.float32:
            # Float64 buffers of the same element count are accepted and
            # downcast; non-float dtypes are reinterpreted byte-wise only
            # when the byte size matches.
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            elif arr.nbytes == expected * 4:
                arr = arr.view(np.float32)
            else:
                raise ValueError(
                    f"cannot interpret buffer of dtype {arr.dtype} as float32 field"
                )
        field = arr.reshape(self.shape)
        if not field.flags["C_CONTIGUOUS"]:
            field = np.ascontiguousarray(field)
        return np.ascontiguousarray(field, dtype=np.float32)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, buf) -> bytes:
        field = self._as_field(buf)
        mask = extract_mask(field, self.missing_value)
        valid_values = field[~mask].astype(np.float32)
        n_valid = int(valid_values.size)

        # --- Prediction: transformer sees only the mask ------------------
        prediction = self._predictor.predict(mask)
        pred_valid = prediction[~mask].astype(np.float32)
        residuals = valid_values.astype(np.float64) - pred_valid.astype(np.float64)

        # --- Quantization + verify-and-repair ----------------------------
        bound = self.error_bound
        repair_positions = np.zeros(0, dtype=np.int64)
        repair_values = np.zeros(0, dtype=np.float32)
        if bound == 0.0:
            # FR-003 exemption: tightest available representation = raw bits.
            step = 0.0
            origin = 0.0
            mode = _MODE_EXACT
            value_payload = valid_values.tobytes()
            max_abs_error = 0.0
        else:
            step = derive_step(bound)
            origin = float(np.mean(residuals)) if n_valid else 0.0
            symbols = quantize(residuals, step, origin)
            # Simulate the exact decode path:
            decoded_valid = (
                pred_valid.astype(np.float64)
                + dequantize(symbols, step, origin)
            ).astype(np.float32)
            repair_map, _, max_abs_error, _ = verify_and_repair(
                valid_values, decoded_valid, bound
            )
            repair_positions, repair_values = (
                repair_map.positions,
                repair_map.values,
            )
            # --- Entropy coding (bit-exact; per-block mode selection) ----
            encoded_syms = _entropy_encode(symbols)
            body = (
                struct.pack("<Q", len(encoded_syms))
                + encoded_syms
                + pack_repairs(repair_positions, repair_values)
            )
            value_payload = body
            mode = _MODE_QUANTIZED

        # --- Mask + container -------------------------------------------
        mask_payload = pack_mask(mask)
        header_extra = {
            "shape": list(self.shape),
            "dtype": "float32",
            "missing_value": "nan" if np.isnan(self.missing_value) else float(self.missing_value),
            "error_bound": float(bound),
            "quant_step": step,
            "origin": float(origin) if bound > 0 else 0.0,
            "mode": mode,
            "n_valid": int(valid_values.size),
            "n_repaired": int(repair_positions.size),
            "metrics": {
                # FR-012: payload-based metrics (stable, non-self-referential).
                "max_abs_error": max_abs_error,
                "n_repaired": int(repair_positions.size),
                "uncompressed_size": int(field.nbytes),
                "payload_size": int(len(mask_payload) + len(value_payload)),
                "bound_respected": bool(max_abs_error <= bound) or bound == 0.0,
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
            expected = int(np.prod(self.shape)) * 4
            if out.nbytes != expected:
                raise ValueError(
                    f"out buffer must be exactly {expected} bytes "
                    f"(shape {self.shape}, float32); got {out.nbytes}"
                )
            if not out.flags["C_CONTIGUOUS"] or out.dtype != np.float32:
                raise ValueError("out buffer must be C-contiguous float32")

        try:
            c = read_container(buf)
        except ContainerError:
            raise
        h = c.header_extra
        shape = tuple(h["shape"])
        if shape != self.shape:
            raise ValueError(
                f"container shape {shape} does not match codec shape {self.shape}"
            )
        step = float(h["quant_step"])
        origin = float(h["origin"])
        mode = int(h.get("mode", 0))
        n_valid = int(h["n_valid"])
        missing_value = h["missing_value"]
        sentinel = float("nan") if missing_value == "nan" else float(missing_value)

        mask = unpack_mask(
            c.mask_payload, int(np.prod(self.shape))
        ).reshape(self.shape)

        if mode == _MODE_EXACT or step == 0.0:
            if n_valid:
                raw = c.residual_payload[: n_valid * 4]
                if len(raw) < n_valid * 4:
                    raise ValueError("exact payload truncated")
                decoded_valid = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                decoded_valid = np.zeros(0, dtype=np.float32)
        else:
            payload = c.residual_payload
            if len(payload) < 8:
                raise ValueError("residual payload too short")
            (ent_len,) = struct.unpack_from("<Q", payload, 0)
            symbols = decode_symbols(payload[8 : 8 + ent_len], n_valid)
            repair_positions, repair_values, _ = unpack_repairs(payload, 8 + ent_len)
            # Prediction identical to encode (mask-driven transformer).
            prediction = self._predictor.predict(mask)
            pred_valid = prediction[~mask].astype(np.float64)[:n_valid]
            decoded_valid = (
                pred_valid + dequantize(symbols[:n_valid], step, origin)
            ).astype(np.float32)
            if repair_positions.size:
                decoded_valid[repair_positions] = repair_values

        field = np.empty(self.shape, dtype=np.float32)
        if n_valid:
            field[~mask] = decoded_valid
        field = apply_mask(decoded=field, mask=mask, missing_value=sentinel)

        if out is not None:
            out[...] = field.reshape(out.shape)
            return out
        return field
