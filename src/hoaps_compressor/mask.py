"""Missing-value mask extraction and bitpacking (FR-004, FR-015).

The mask is one bit per grid element (True = missing), bitpacked into
``ceil(N / 8)`` bytes and stored losslessly in its own container payload.
"""

from __future__ import annotations

import numpy as np


def extract_mask(values: np.ndarray, missing_value) -> np.ndarray:
    """Return a bool array where True marks missing elements.

    ``missing_value`` may be a finite float or the string ``"nan"``.
    Elements equal to the sentinel are missing; when the sentinel is NaN,
    NaN elements are missing. Any non-finite element that does not match a
    finite sentinel is also treated as missing (FR-011).
    """
    values = np.asarray(values, dtype=np.float32)
    if isinstance(missing_value, str) and missing_value.lower() == "nan":
        # NaN sentinel: NaN and any other non-finite value count as missing
        return ~np.isfinite(values)
    missing_value = float(missing_value)
    if np.isnan(missing_value):
        return ~np.isfinite(values)
    mask = values == np.float32(missing_value)
    # FR-011: non-finite values not matching the sentinel -> missing
    mask |= ~np.isfinite(values)
    return mask


def pack_mask(mask: np.ndarray) -> bytes:
    """Bitpack a bool mask (row-major) into minimal bytes (ceil(n/8))."""
    flat = np.asarray(mask, dtype=bool).ravel()
    n = flat.size
    if n == 0:
        return b""
    packed = np.packbits(flat, bitorder="little")
    return packed.tobytes()


def unpack_mask(packed: bytes, n: int) -> np.ndarray:
    """Unpack ``n`` mask bits from ``packed`` bytes; bit-exact inverse."""
    if n == 0:
        return np.zeros(0, dtype=bool)
    expected = (n + 7) // 8
    buf = bytes(packed)
    if len(buf) != expected:
        raise ValueError(
            f"mask payload length mismatch: expected {expected} bytes for "
            f"{n} elements, got {len(buf)}"
        )
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=n, bitorder="little")
    return bits.astype(bool)


def apply_mask(decoded: np.ndarray, mask: np.ndarray, missing_value: float) -> np.ndarray:
    """Return decoded values with the sentinel restored at missing positions."""
    out = np.asarray(decoded, dtype=np.float32).copy()
    out[mask] = np.float32(missing_value)
    return out.astype(np.float32)


def mask_is_identical(original: np.ndarray, reconstructed: np.ndarray) -> bool:
    """SC-002 check: missing masks of two fields (or a field + sentinel) are identical."""
    ref_missing = np.isnan(original) if original.dtype.kind == "f" else None
    rec_missing = np.isnan(reconstructed) if reconstructed.dtype.kind == "f" else None
    if ref_missing is None or rec_missing is None:
        return False
    return bool(np.array_equal(ref_missing, rec_missing))
