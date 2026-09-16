"""Shared pytest fixtures: deterministic BRACE-like synthetic fields."""

import numpy as np
import pytest

# Default test grid shape: (time, lat, lon)
DEFAULT_SHAPE = (4, 16, 32)
DEFAULT_BOUND = 0.05


def make_smooth_field(shape=DEFAULT_SHAPE, seed=42, valid_frac=0.85, land_strip=True):
    """Generate a smooth space-time field with a deterministic missing mask.

    Values use a smooth climate-field scale (~0-60 kg/m2) with slow evolution
    in latitude and longitude. A fraction ``valid_frac`` of cells is valid; the
    rest are missing (NaN sentinel). Set ``land_strip=False`` for a
    fully-valid field.
    """
    rng = np.random.default_rng(seed)
    t, lat, lon = shape
    x = np.linspace(0.0, 2.0 * np.pi, lon)
    y = np.linspace(0.0, np.pi, lat)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    field = np.empty((t, lat, lon), dtype=np.float32)
    for ti in range(t):
        phase = 0.35 * ti
        base = 25.0 + 12.0 * np.sin(yy + phase) * np.cos(xx - 0.5 * phase)
        detail = rng.normal(0.0, 0.8, size=(lat, lon))
        field[ti] = (base + detail).astype(np.float32)

    # Deterministic "land" strip mask + scattered missing cells
    mask = np.zeros(shape, dtype=bool)
    if land_strip:
        mask[:, : max(1, lat // 8), :] = True
    if valid_frac < 1.0:
        scatter = rng.random(shape) > valid_frac
        mask |= scatter

    return np.where(mask, np.nan, field).astype(np.float32), mask


@pytest.fixture
def smooth_field():
    field, mask = make_smooth_field()
    return field, mask


@pytest.fixture
def codec_factory():
    """Factory to build codecs with defaults matching DEFAULT_SHAPE."""
    from brace_compressor import BraceCodec

    def _make(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND, **kw):
        return BraceCodec(shape=shape, error_bound=error_bound, **kw)

    return _make
