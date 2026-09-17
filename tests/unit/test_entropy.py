"""Unit tests for the bit-exact entropy coder."""

import numpy as np
import pytest

from brace_compressor.model.entropy import (
    MODE_CTX,
    MODE_RANGE,
    MODE_RAW,
    _HEAD,
    _RAWLEN,
    ctx_rans_decode,
    ctx_rans_encode,
    decode_symbols,
    encode_symbols,
    pack_repairs,
    unpack_repairs,
    unzigzag,
    varint_decode,
    varint_encode,
    zigzag,
)


@pytest.mark.parametrize(
    "symbols",
    [
        np.zeros(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1, -1, 2, -2, 0], dtype=np.int64),
        np.arange(-5000, 5000, dtype=np.int64),
        (np.arange(5000, dtype=np.int64) % 20001) - 10000,
        np.array([-(2**40), 2**40, 0], dtype=np.int64),  # forces 8-byte path
        (np.random.default_rng(0).normal(0, 1000, 5000)).astype(np.int64),
    ],
)
def test_symbols_roundtrip_bitexact(symbols):
    payload = encode_symbols(symbols)
    out = decode_symbols(payload, symbols.size)
    np.testing.assert_array_equal(out, symbols)


def test_symbols_all_modes_covered():
    # Small values -> 1-byte raw or range; huge values -> 8-byte raw
    tiny = np.arange(-100, 100, dtype=np.int64)
    huge = np.array([-(2**62), 2**62], dtype=np.int64)
    for s in (tiny, huge):
        np.testing.assert_array_equal(decode_symbols(encode_symbols(s), s.size), s)


def test_ctx_rans_roundtrip():
    """Context-adaptive rANS must round-trip bit-exactly."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        # Smooth-ish symbol stream: small residuals with occasional jumps
        n = 3000
        base = rng.normal(0, 2, n).astype(np.int64)
        jumps = (rng.random(n) < 0.02) * rng.integers(-200, 200, n)
        syms = base + jumps
        span = int(syms.max()) - int(syms.min()) + 1
        min_symbol = int(syms.min())
        idx = (syms - min_symbol).astype(np.int64)
        ctxs = np.zeros(n, dtype=np.int64)
        a = np.abs(syms[:-1])
        ctxs[1:] = np.where(a <= 1, 0, np.where(a <= 8, 1, 2))
        counts = np.zeros((3, span), dtype=np.int64)
        np.add.at(counts, (ctxs, idx), 1)
        from brace_compressor.model.entropy import _normalize_freqs_span

        freqs_list = [_normalize_freqs_span(counts[c], span) for c in range(3)]
        stream = ctx_rans_encode(syms, freqs_list, min_symbol)
        out = ctx_rans_decode(stream, n, freqs_list, min_symbol)
        np.testing.assert_array_equal(out, syms)


def test_ctx_mode_selected_when_smaller():
    """CTX mode must be selected when it beats the RANGE payload."""
    rng = np.random.default_rng(1)
    # Smooth residual stream: strong conditional structure
    n = 5000
    syms = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        syms[i] = syms[i - 1] + rng.integers(-2, 3)
    payload = encode_symbols(syms)
    out = decode_symbols(payload, n)
    np.testing.assert_array_equal(out, syms)
    # Verify the mode bytes say CTX
    n_blocks = (n + 2047) // 2048
    off = _HEAD.size
    (nb,) = _HEAD.unpack_from(payload, 0)
    assert nb == n_blocks
    mode_bytes = payload[off : off + nb]
    assert all((m & 0x0F) == MODE_CTX for m in mode_bytes)


def test_ctx_falls_back_to_range_for_large_span():
    """CTX must fall back to RANGE when the symbol alphabet is too large."""
    rng = np.random.default_rng(2)
    # Wide-span symbols: CTX tables would be huge
    syms = rng.integers(-(2**20), 2**20, 3000).astype(np.int64)
    payload = encode_symbols(syms)
    out = decode_symbols(payload, syms.size)
    np.testing.assert_array_equal(out, syms)
    # Mode bytes must NOT be all CTX
    n_blocks = (syms.size + 2047) // 2048
    off = _HEAD.size
    (nb,) = _HEAD.unpack_from(payload, 0)
    mode_bytes = payload[off : off + nb]
    assert not all((m & 0x0F) == MODE_CTX for m in mode_bytes)


def test_varint_roundtrip():
    vals = np.array([0, 1, 127, 128, 300, 2**20, 2**35, 2**63 - 1], dtype=np.uint64)
    enc = varint_encode(vals)
    np.testing.assert_array_equal(varint_decode(enc, vals.size), vals)


def test_zigzag_roundtrip():
    vals = np.array([0, -1, 1, -2, 2, -(2**62), 2**62], dtype=np.int64)
    np.testing.assert_array_equal(unzigzag(zigzag(vals)), vals)


def test_repairs_roundtrip():
    pos = np.array([3, 17, 999], dtype=np.int64)
    val = np.array([1.5, -2.25, 33.0], dtype=np.float32)
    payload = encode_symbols(np.array([0, 5], dtype=np.int64)) + pack_repairs(pos, val)
    total = payload
    syms = decode_symbols(total[: -len(pack_repairs(pos, val))], 2)
    np.testing.assert_array_equal(syms, [0, 5])
    p2, v2, end = unpack_repairs(total, len(total) - len(pack_repairs(pos, val)))
    np.testing.assert_array_equal(p2, pos)
    np.testing.assert_array_equal(v2, val)
    assert end == len(total)


def test_float64_repairs_roundtrip():
    pos = np.array([3, 17], dtype=np.int64)
    val = np.array([1.234567890123, -9876543.210987], dtype=np.float64)
    payload = pack_repairs(pos, val, dtype=np.float64)

    p2, v2, end = unpack_repairs(payload, dtype=np.float64)

    np.testing.assert_array_equal(p2, pos)
    np.testing.assert_array_equal(v2, val)
    assert v2.dtype == np.float64
    assert end == len(payload)


def test_repair_section_offset():
    syms = np.arange(5000, dtype=np.int64) - 2500
    pos = np.array([7], dtype=np.int64)
    val = np.array([9.0], dtype=np.float32)
    body = encode_symbols(syms)
    payload = body + pack_repairs(pos, val)
    out = decode_symbols(payload[: len(body)], syms.size)
    np.testing.assert_array_equal(out, syms)
    p, v, end = unpack_repairs(payload, len(body))
    np.testing.assert_array_equal(p, pos)
    np.testing.assert_array_equal(v, val)
    assert end == len(payload)
