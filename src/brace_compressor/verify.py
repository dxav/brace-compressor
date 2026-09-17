"""Verify-and-repair loop for the positive-bound path.

After a candidate encoding, the encoder simulates the exact decode path,
measures per-element error against the bound, and repairs any violating
elements by escalating precision for them (exact float32 correction).
The bound thus cannot ship violated beyond the effective bound. Bounds at or
below float32 epsilon use that epsilon as the effective budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RepairMap:
    """Per-element exact corrections for bound-violating elements."""

    positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __bool__(self) -> bool:
        return self.positions.size > 0

    def __len__(self) -> int:
        return int(self.positions.size)


def measure_errors(
    orig_valid: np.ndarray, decoded_valid: np.ndarray, dtype=np.float32
) -> np.ndarray:
    """Per-element absolute error between original and reconstructed values."""
    value_dtype = np.dtype(dtype)
    orig_valid = np.asarray(orig_valid, dtype=value_dtype)
    decoded_valid = np.asarray(decoded_valid, dtype=value_dtype)
    if orig_valid.shape != decoded_valid.shape:
        raise ValueError(
            f"shape mismatch between original ({orig_valid.shape}) and "
            f"decoded ({decoded_valid.shape})"
        )
    return np.abs(
        orig_valid.astype(np.float64) - decoded_valid.astype(np.float64)
    )


def violations(
    orig_valid, decoded_valid, error_bound: float, dtype=np.float32
) -> np.ndarray:
    """Indices (into the valid-value list) of elements exceeding the bound."""
    errors = measure_errors(orig_valid, decoded_valid, dtype=dtype)
    return np.nonzero(errors > np.asarray(error_bound, dtype=dtype))[0].astype(
        np.int64
    )


def verify_and_repair(
    orig_valid, decoded_valid, error_bound, repair_fn=None, dtype=np.float32
):
    """Verify decoded values against the bound; repair violations by escalation.

    Args:
        orig_valid: original valid values (float32 array).
        decoded_valid: values reconstructed exactly as the decoder would.
        error_bound: effective positive absolute error bound, including the
            float32-epsilon floor used for a requested zero bound.
        repair_fn: optional callables registry, unused in v1 (kept for
            forward-compatible escalation hooks).

    Returns:
        tuple (repair_map, residual_errors, max_abs_error, violations_fixed):
        repair_map lists exact positions+values so the decoder can apply
        them; residual_errors is the post-repair per-element error array;
        ``max_abs_error`` is the final verified maximum (<= bound).
    """
    value_dtype = np.dtype(dtype)
    errors = measure_errors(orig_valid, decoded_valid, dtype=value_dtype)
    bad = np.nonzero(errors > np.asarray(error_bound, dtype=value_dtype))[0].astype(
        np.int64
    )

    if bad.size == 0:
        max_abs = float(errors.max()) if errors.size else 0.0
        return RepairMap(), errors, max_abs, 0

    # Escalation: exact correction in the configured field dtype.
    reconstruction = np.asarray(decoded_valid, dtype=value_dtype).copy()
    reconstruction[bad] = np.asarray(orig_valid, dtype=value_dtype)[bad]
    repair_map = RepairMap(positions=bad, values=orig_valid[bad].astype(value_dtype))

    post = measure_errors(
        np.asarray(orig_valid, dtype=value_dtype), reconstruction, dtype=value_dtype
    )
    remaining = np.nonzero(post > np.asarray(error_bound, dtype=value_dtype))[0]
    if remaining.size:
        # Exact corrections can still be limited by representation precision.
        max_abs = float(post.max())
    else:
        max_abs = float(post.max()) if post.size else 0.0

    return repair_map, post, max_abs, int(bad.size)
