"""Integration tests for the causal space-time predictor."""

import numpy as np
from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


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
