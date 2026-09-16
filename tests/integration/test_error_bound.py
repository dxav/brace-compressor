"""Integration tests for error bounds and compression ratio."""

import numpy as np
import pytest

from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


@pytest.mark.parametrize("bound", [0.01, 0.05, 0.2])
def test_roundtrip_respects_bound(bound):
    field, _ = make_smooth_field()
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=bound)
    enc = codec.encode(field)
    dec = codec.decode(enc)
    valid = ~np.isnan(field)
    err = np.abs((dec - field)[valid])
    assert err.max() <= bound  # SC-001 / FR-003
    assert field.nbytes / len(enc) > 1.0  # FR-006: smaller than original


def test_cr_at_least_50pct_smaller():
    """SC-003 on a typical-scale field (SC-003 says 'typical HOAPS wvpa inputs')."""
    # Typical-scale field (1.5 MB class, far above tiny toy grids where the
    # fixed container header dominates):
    shape = (8, 90, 180)
    field, _ = make_smooth_field(shape=shape)
    codec = HoapsWvpaCodec(shape=shape, error_bound=DEFAULT_BOUND)
    enc = codec.encode(field)
    assert len(enc) <= 0.5 * field.nbytes  # SC-003
    # metrics recorded in header (FR-012)
    from hoaps_compressor.container import read_container

    meta = read_container(enc).header_extra["metrics"]
    assert meta["max_abs_error"] <= DEFAULT_BOUND
    assert meta["payload_size"] <= field.nbytes  # payloads beat raw storage
    assert meta["uncompressed_size"] == field.nbytes
    assert meta["bound_respected"] is True
    # small toy fields still round-trip and do not grow unreasonably
    small, _ = make_smooth_field()
    small_codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    small_enc = small_codec.encode(small)
    assert len(small_enc) < 4 * small.nbytes


def test_metrics_reported():
    """FR-012: encode reports metrics and bound confirmation."""
    field, _ = make_smooth_field()
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    enc = codec.encode(field)
    from hoaps_compressor.container import read_container

    m = read_container(enc).header_extra["metrics"]
    assert m["max_abs_error"] <= DEFAULT_BOUND
    assert m["n_repaired"] >= 0
    assert m["uncompressed_size"] == field.nbytes
    assert 0 < m["payload_size"] <= field.nbytes
    assert m["bound_respected"] is True


def test_bound_zero_is_tightest_and_may_be_lossy():
    field, _ = make_smooth_field()
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=0.0)
    enc = codec.encode(field)
    dec = codec.decode(enc)
    # FR-003 exemption: output may be lossy; it must simply decode validly.
    assert dec.shape == DEFAULT_SHAPE
    # Bound=0 produces the tightest available representation; smaller than
    # the tightest *quantized* bound-0.01 payload is NOT required, but the
    # mask must still be exact.
    assert np.array_equal(np.isnan(dec), np.isnan(field))


def test_decode_single_element_grid():
    field = np.array([[[42.5]]], dtype=np.float32)
    codec = HoapsWvpaCodec(shape=(1, 1, 1), error_bound=0.01)
    dec = codec.decode(codec.encode(field))
    assert abs(float(dec[0, 0, 0]) - 42.5) <= 0.01
