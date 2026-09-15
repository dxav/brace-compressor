"""Residual quantization bound-tied to the absolute error bound (T007, FR-016).

For bound > 0, the step ``Δ = 2·bound`` gives per-element quantization error
``<= Δ/2 = bound`` — the full error budget. Because the predictor is applied
symmetrically at encode and decode, its error cancels out of the
reconstruction, so the total error is purely the residual's quantization
error (<= bound). verify-and-repair remains as a safety net for rare
floating-point rounding that could push a value a hair over the bound.
For bound == 0 (fr-003-exempt), the "tightest available representation" is
used: exact float32 bit patterns.
"""

from __future__ import annotations

import numpy as np


def derive_step(error_bound: float) -> float:
    """Derive quantization step Δ from the absolute error bound.

    ``Δ = 2·bound`` so quantization error <= Δ/2 = bound, using the full
    error budget and maximizing CR. The predictor error cancels because it
    is applied identically at encode and decode. Raises ValueError for
    non-finite/negative bounds (defensive; bound validation normally
    happens in bound.py).
    """
    import math

    if math.isnan(error_bound) or math.isinf(error_bound) or error_bound < 0:
        raise ValueError(f"invalid error_bound: {error_bound!r}")
    if error_bound == 0.0:
        return 0.0  # bound=0: no quantization; exact path (FR-003 exemption)
    return 2.0 * float(error_bound)


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
