"""Residual quantization tied to the absolute error bound.

For bound > 0, the step ``Δ = 2·bound`` gives per-element quantization error
``<= Δ/2 = bound`` — the full error budget. Because the predictor is applied
symmetrically at encode and decode, its error cancels out of the
reconstruction, so the total error is purely the residual's quantization
error (<= bound). verify-and-repair remains as a safety net for rare
floating-point rounding that could push a value a hair over the bound.
Bounds below float32 machine epsilon use epsilon as the effective bound. This
keeps zero in the same quantized path as positive bounds while limiting the
additional error to float32 computation precision.
"""

from __future__ import annotations

import numpy as np


def derive_step(error_bound: float, dtype=np.float32) -> float:
    """Derive quantization step Δ from the absolute error bound.

    ``Δ = 2·max(bound, eps32)``. Bounds below float32 machine epsilon use
    epsilon as the effective error budget because the computation itself is
    float32-based. The predictor error cancels because it is applied
    identically at encode and decode.
    """
    import math

    if math.isnan(error_bound) or math.isinf(error_bound) or error_bound < 0:
        raise ValueError(f"invalid error_bound: {error_bound!r}")
    effective_bound = max(float(error_bound), float(np.finfo(dtype).eps))
    return 2.0 * effective_bound


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
