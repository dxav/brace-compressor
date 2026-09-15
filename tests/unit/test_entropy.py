"""Unit tests for the bit-exact entropy coder (T028, US4)."""

import numpy as np
import pytest

from hoaps_compressor.model.entropy import (
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
