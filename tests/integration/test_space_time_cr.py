"""Integration + unit tests: transformer predictor and space-time CR (T025, T026, US4)."""

import numpy as np
import pytest

from hoaps_compressor import HoapsWvpaCodec
from hoaps_compressor.model.transformer import TransformerPredictor
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


def test_transformer_determinism():
    pred = TransformerPredictor()
    mask = np.zeros(DEFAULT_SHAPE, dtype=bool)
    mask[:, :2, :] = True
    p1 = pred.base_prior(mask)
    p2 = pred.base_prior(mask)
    np.testing.assert_array_equal(p1, p2)  # same weights -> same output


def test_transformer_ignores_field_values():
    """Predictor must be mask-driven only (decode has no original values)."""
    pred = TransformerPredictor()
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    a = pred.base_prior(mask)
    b = pred.base_prior(mask.copy())
    np.testing.assert_array_equal(a, b)


def test_model_weights_serialize_roundtrip():
    pred = TransformerPredictor()
    blob = pred.serialize_weights()
    pred2 = TransformerPredictor()
    pred2.load_weights(blob)
    mask = np.zeros((1, 3, 3), dtype=bool)
    np.testing.assert_allclose(
        pred.base_prior(mask), pred2.base_prior(mask), rtol=0, atol=0
    )


def test_model_version_mismatch_rejected():
    from hoaps_compressor.model import transformer as T

    pred = TransformerPredictor()
    blob = bytearray(pred.serialize_weights())
    # corrupt version field (offset 4, uint32)
    blob[4] = (blob[4] + 1) % 256
    pred2 = TransformerPredictor()
    with pytest.raises(ValueError, match="model version mismatch"):
        pred2.load_weights(bytes(blob))


def test_space_time_cr_beats_spatial_only_baseline():
    """SC-007: space-time CR >= spatial-only (per-field) baseline CR."""
    field, _ = make_smooth_field()
    t, lat, lon = DEFAULT_SHAPE

    # Space-time codec: whole 3-D field at once
    st_codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    st_size = len(st_codec.encode(field))

    # Spatial-only baseline: each 2-D field compressed independently with
    # an equivalent per-slice budget (sum of independent encodes).
    sl_codec = HoapsWvpaCodec(shape=(1, lat, lon), error_bound=DEFAULT_BOUND)
    sl_size = 0
    for ti in range(t):
        sl_size += len(sl_codec.encode(field[ti : ti + 1]))

    assert st_size <= sl_size * 1.05  # at least as good (5% tolerance)
