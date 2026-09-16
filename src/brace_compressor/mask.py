"""Missing-value mask extraction and compact lossless encoding (FR-004, FR-015).

The mask is one bit per grid element (True = missing). Structured masks use a
self-describing run-length encoding; less structured masks use bitpacking.
"""

from __future__ import annotations

import numpy as np


_RLE_MAGIC = b"HMR1"


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 128:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _pack_rle(flat: np.ndarray) -> bytes:
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    lengths = np.diff(boundaries)
    out = bytearray(_RLE_MAGIC)
    out.append(int(flat[0]))
    for length in lengths:
        out.extend(_encode_varint(int(length)))
    return bytes(out)


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
    """Encode a mask as compact RLE or legacy-compatible bitpacked bytes."""
    flat = np.asarray(mask, dtype=bool).ravel()
    n = flat.size
    if n == 0:
        return b""
    packed = np.packbits(flat, bitorder="little")
    rle = _pack_rle(flat)
    return rle if len(rle) < len(packed) else packed.tobytes()


def _unpack_rle(packed: bytes, n: int) -> np.ndarray:
    data = memoryview(packed)
    if len(data) < len(_RLE_MAGIC) + 1:
        raise ValueError("truncated RLE mask payload")
    value = bool(data[len(_RLE_MAGIC)])
    offset = len(_RLE_MAGIC) + 1
    flat = np.empty(n, dtype=bool)
    position = 0
    while position < n:
        length = 0
        shift = 0
        while True:
            if offset >= len(data) or shift > 63:
                raise ValueError("invalid RLE mask varint")
            byte = int(data[offset])
            offset += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        if length <= 0 or position + length > n:
            raise ValueError("RLE mask run exceeds expected length")
        flat[position : position + length] = value
        position += length
        value = not value
    if offset != len(data):
        raise ValueError("RLE mask payload has trailing bytes")
    return flat


def unpack_mask(packed: bytes, n: int) -> np.ndarray:
    """Unpack optimized RLE or legacy bitpacked mask bytes."""
    if n == 0:
        return np.zeros(0, dtype=bool)
    if bytes(packed).startswith(_RLE_MAGIC):
        return _unpack_rle(bytes(packed), n)
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
