"""Config contract tests (T020, US3)."""

import json

import pytest

from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE


def test_get_config_keys():
    cfg = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND).get_config()
    assert set(cfg.keys()) == {
        "id",
        "shape",
        "error_bound",
        "missing_value",
        "dtype",
        "outer_compress",
    }
    assert cfg["id"] == "hoaps-wvpa"
    assert cfg["shape"] == list(DEFAULT_SHAPE)
    assert cfg["error_bound"] == DEFAULT_BOUND
    assert cfg["missing_value"] == "nan"
    assert cfg["dtype"] == "float32"


def test_config_json_serializable():
    cfg = HoapsWvpaCodec(
        shape=(1, 2, 3), error_bound=0.2, missing_value=-999.0, outer_compress=False
    ).get_config()
    restored = json.loads(json.dumps(cfg))
    assert restored == cfg


def test_from_config_full_cycle():
    original = HoapsWvpaCodec(
        shape=(2, 3, 4), error_bound=0.25, missing_value=-32767.0
    )
    rebuilt = HoapsWvpaCodec.from_config(original.get_config())
    assert rebuilt.shape == original.shape
    assert rebuilt.error_bound == original.error_bound
    assert rebuilt.missing_value == original.missing_value


def test_from_config_missing_keys():
    with pytest.raises(ValueError, match="required config key"):
        HoapsWvpaCodec.from_config({"id": "hoaps-wvpa"})
