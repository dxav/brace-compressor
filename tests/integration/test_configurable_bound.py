"""Integration tests: configurable error bound (T021, US3)."""

import numpy as np
import pytest

from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_SHAPE, make_smooth_field


def test_cr_monotonic_in_bound():
    field, _ = make_smooth_field()
    sizes = []
    for b in (0.01, 0.05, 0.2):
        codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=b)
        enc = codec.encode(field)
        sizes.append(len(enc))
        dec = codec.decode(enc)
        valid = ~np.isnan(field)
        assert np.abs((dec - field)[valid]).max() <= b  # each bound honored
    # SC-004: looser bound -> smaller (or equal) size
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_cr_monotonic_typical_scale():
    """SC-003/SC-004 at typical scale: reference bound >= 50% reduction."""
    shape = (8, 90, 180)
    field, _ = make_smooth_field(shape=shape)
    sizes = {}
    for b in (0.01, 0.05, 0.2):
        codec = HoapsWvpaCodec(shape=shape, error_bound=b)
        enc = codec.encode(field)
        sizes[b] = len(enc)
        dec = codec.decode(enc)
        valid = ~np.isnan(field)
        assert np.abs((dec - field)[valid]).max() <= b
    assert sizes[0.01] >= sizes[0.05] >= sizes[0.2]
    assert sizes[0.05] <= 0.5 * field.nbytes  # SC-003


def test_config_roundtrip_behavioral_identity():
    field, _ = make_smooth_field()
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=0.05)
    cfg = codec.get_config()
    codec2 = HoapsWvpaCodec.from_config(cfg)
    e1 = codec.encode(field)
    e2 = codec2.encode(field)
    assert e1 == e2  # byte-identical encodes
    np.testing.assert_array_equal(codec2.decode(e1), codec.decode(e1))


def test_invalid_bounds_rejected():
    for bad in (-0.001, float("-inf"), float("inf"), float("nan")):
        with pytest.raises(ValueError):
            HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=bad)
