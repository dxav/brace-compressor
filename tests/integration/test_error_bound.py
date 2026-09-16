"""Integration tests for error bounds and compression ratio."""

import numpy as np
import pytest

from brace_compressor import BraceCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


@pytest.mark.parametrize("bound", [0.01, 0.05, 0.2])
def test_roundtrip_respects_bound(bound):
    field, _ = make_smooth_field()
    codec = BraceCodec(shape=DEFAULT_SHAPE, error_bound=bound)
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
    codec = BraceCodec(shape=shape, error_bound=DEFAULT_BOUND)
    enc = codec.encode(field)
    assert len(enc) <= 0.5 * field.nbytes  # SC-003
    # metrics recorded in header (FR-012)
    from brace_compressor.container import read_container

    meta = read_container(enc).header_extra["metrics"]
    assert meta["max_abs_error"] <= DEFAULT_BOUND
    assert meta["payload_size"] <= field.nbytes  # payloads beat raw storage
    assert meta["uncompressed_size"] == field.nbytes
    assert meta["bound_respected"] is True
    # small toy fields still round-trip and do not grow unreasonably
    small, _ = make_smooth_field()
    small_codec = BraceCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    small_enc = small_codec.encode(small)
    assert len(small_enc) < 4 * small.nbytes


def test_metrics_reported():
    """FR-012: encode reports metrics and bound confirmation."""
    field, _ = make_smooth_field()
    codec = BraceCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    enc = codec.encode(field)
    from brace_compressor.container import read_container

    m = read_container(enc).header_extra["metrics"]
    assert m["max_abs_error"] <= DEFAULT_BOUND
    assert m["n_repaired"] >= 0
    assert m["uncompressed_size"] == field.nbytes
    assert 0 < m["payload_size"] <= field.nbytes
    assert m["bound_respected"] is True


def test_bound_zero_uses_near_lossless_epsilon_mode():
    field, _ = make_smooth_field()
    codec = BraceCodec(shape=DEFAULT_SHAPE, error_bound=0.0)
    enc = codec.encode(field)
    dec = codec.decode(enc)
    assert dec.shape == DEFAULT_SHAPE
    assert np.array_equal(np.isnan(dec), np.isnan(field))
    valid = ~np.isnan(field)
    effective_bound = np.finfo(np.float32).eps
    assert np.max(np.abs(dec[valid] - field[valid])) <= effective_bound


def test_decode_single_element_grid():
    field = np.array([[[42.5]]], dtype=np.float32)
    codec = BraceCodec(shape=(1, 1, 1), error_bound=0.01)
    dec = codec.decode(codec.encode(field))
    assert abs(float(dec[0, 0, 0]) - 42.5) <= 0.01
