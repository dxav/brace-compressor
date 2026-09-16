"""Configuration contract tests."""

import json

import pytest

from brace_compressor import BraceCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE


def test_get_config_keys():
    cfg = BraceCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND).get_config()
    assert set(cfg.keys()) == {
        "id",
        "shape",
        "error_bound",
        "missing_value",
        "dtype",
        "outer_compress",
    }
    assert cfg["id"] == "brace-wvpa"
    assert cfg["shape"] == list(DEFAULT_SHAPE)
    assert cfg["error_bound"] == DEFAULT_BOUND
    assert cfg["missing_value"] == "nan"
    assert cfg["dtype"] == "float32"


def test_config_json_serializable():
    cfg = BraceCodec(
        shape=(1, 2, 3), error_bound=0.2, missing_value=-999.0, outer_compress=False
    ).get_config()
    restored = json.loads(json.dumps(cfg))
    assert restored == cfg


def test_from_config_full_cycle():
    original = BraceCodec(
        shape=(2, 3, 4), error_bound=0.25, missing_value=-32767.0
    )
    rebuilt = BraceCodec.from_config(original.get_config())
    assert rebuilt.shape == original.shape
    assert rebuilt.error_bound == original.error_bound
    assert rebuilt.missing_value == original.missing_value


def test_from_config_missing_keys():
    with pytest.raises(ValueError, match="required config key"):
        BraceCodec.from_config({"id": "brace-wvpa"})
