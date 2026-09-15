"""Residual quantization bound-tied to the absolute error bound (T007, FR-016).

For bound > 0, the step ``Δ <= bound`` guarantees per-element quantization
error ``<= Δ/2 <= bound/2``, leaving headroom for predictor error and the
verify-and-repair stage. For bound == 0 (fr-003-exempt), the "tightest
available representation" is used: exact float32 bit patterns.
"""

from __future__ import annotations

import numpy as np


def derive_step(error_bound: float) -> float:
    """Derive quantization step Δ from the absolute error bound.

    ``Δ = bound / 2`` so quantization error <= Δ/2 = bound/4, leaving
    half the bound as predictor-error headroom for verify-and-repair.
    Raises ValueError for non-finite/negative bounds (defensive; bound
    validation normally happens in bound.py).
    """
    import math

    if math.isnan(error_bound) or math.isinf(error_bound) or error_bound < 0:
        raise ValueError(f"invalid error_bound: {error_bound!r}")
    if error_bound == 0.0:
        return 0.0  # bound=0: no quantization; exact path (FR-003 exemption)
    return float(error_bound) / 2.0


def quantize(residual: np.ndarray, step: float, origin: float = 0.0) -> np.ndarray:
    """Quantize residuals to signed integer symbols on the Δ-lattice.

    Full symmetric range of int64 is avoided; values are shifted to
    non-negative and offset by -2**31 so int32-safe symmetry is kept.
    """
    if step == 0.0:
        raise ValueError("quantize() with step=0 is not supported; use exact path")
    residual = np.asarray(residual, dtype=np.float64)
    q = np.floor((residual - origin) / step + 0.5)
    # Clamp to a symmetric int32-safe range
    info = np.iinfo(np.int32)
    q = np.clip(q, info.min + 1, info.max)
    return (q.astype(np.int64) - (1 << 31)).astype(np.int64)


def dequantize(symbols: np.ndarray, step: float, origin: float = 0.0) -> np.ndarray:
    """Reconstruct residual values from integer symbols (inverse of quantize)."""
    if step == 0.0:
        raise ValueError("dequantize() with step=0 is not supported; use exact path")
    return (np.asarray(symbols, dtype=np.int64) + (1 << 31)).astype(
        np.float64
    ) * step + origin


def quantization_error_bound(step: float) -> float:
    """Maximum per-element quantization error (half a step)."""
    return step / 2.0


def exact_encode_values(values: np.ndarray) -> np.ndarray:
    """Bound == 0 path: raw float32 bit patterns (tightest representation)."""
    return np.asarray(values, dtype=np.float32).tobytes()
