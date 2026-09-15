"""Integration tests: missing-value preservation (T016, US2)."""

import numpy as np
import pytest

from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


@pytest.mark.parametrize("seed", [1, 7, 123])
def test_mask_identity_roundtrip(seed):
    field, mask = make_smooth_field(seed=seed)
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    dec = codec.decode(codec.encode(field))
    # SC-002: mask identical, no valid<->missing conversion
    assert np.array_equal(np.isnan(dec), np.isnan(field))
    sent_mask = ~np.isnan(field)
    assert np.array_equal(~np.isnan(dec), sent_mask)


def test_no_missing_values():
    field, _ = make_smooth_field(valid_frac=1.0, land_strip=False)
    assert not np.isnan(field).any()
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    dec = codec.decode(codec.encode(field))
    assert not np.isnan(dec).any()  # no valid converted to missing
    assert np.abs(dec - field).max() <= DEFAULT_BOUND


def test_all_missing():
    am = np.full(DEFAULT_SHAPE, np.nan, dtype=np.float32)
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    enc = codec.encode(am)
    dec = codec.decode(enc)
    assert np.isnan(dec).all()  # FR-009: mask exact, no failure
    assert isinstance(enc, (bytes, bytearray))


def test_finite_sentinel_missing_value():
    field, mask = make_smooth_field()
    sentinel = -999.0
    field_s = np.where(mask, sentinel, field).astype(np.float32)
    codec = HoapsWvpaCodec(
        shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND, missing_value=sentinel
    )
    dec = codec.decode(codec.encode(field_s))
    assert np.array_equal(dec == sentinel, field_s == sentinel)  # mask exact
    valid = field_s != sentinel
    assert np.abs((dec - field_s)[valid]).max() <= DEFAULT_BOUND


def test_stray_nonfinite_treated_as_missing():
    field, _ = make_smooth_field()
    field = field.copy()
    # inject +inf where value was valid (FR-011: non-finite -> missing)
    idx = np.unravel_index(5, field.shape)
    if not np.isnan(field[idx]):
        field[idx] = np.inf
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    dec = codec.decode(codec.encode(field))
    assert ~np.isfinite(dec[idx])  # stays non-finite (missing)
