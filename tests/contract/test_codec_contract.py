"""Tests for the public numcodecs.Codec contract."""

import json

import numpy as np
import pytest

import hoaps_compressor  # registers codec on import
from hoaps_compressor import HoapsWvpaCodec
from tests.conftest import DEFAULT_BOUND, DEFAULT_SHAPE, make_smooth_field


def test_codec_id():
    assert HoapsWvpaCodec.codec_id == "hoaps-wvpa"
    assert isinstance(HoapsWvpaCodec.codec_id, str)


def test_registered_in_numcodecs_registry():
    import numcodecs.registry

    codec = numcodecs.registry.get_codec(
        {"id": "hoaps-wvpa", "shape": list(DEFAULT_SHAPE), "error_bound": DEFAULT_BOUND}
    )
    assert isinstance(codec, HoapsWvpaCodec)


def test_get_config_json_roundtrip():
    codec = HoapsWvpaCodec(shape=DEFAULT_SHAPE, error_bound=DEFAULT_BOUND)
    cfg = codec.get_config()
    assert cfg["id"] == "hoaps-wvpa"
    # JSON-serializable (numcodecs contract)
    restored = json.loads(json.dumps(cfg))
    assert restored == cfg
    codec2 = HoapsWvpaCodec.from_config(restored)
    assert codec2.shape == codec.shape
    assert codec2.error_bound == codec.error_bound


def test_from_config_rejects_wrong_id():
    with pytest.raises(ValueError, match="id mismatch"):
        HoapsWvpaCodec.from_config({"id": "other", "shape": [1, 2, 3], "error_bound": 0.1})


def test_encode_decode_buffer_contract(codec_factory, smooth_field):
    field, _ = smooth_field
    codec = codec_factory()
    enc = codec.encode(field)
    assert isinstance(enc, (bytes, bytearray, memoryview))

    dec = codec.decode(enc)
    assert isinstance(dec, np.ndarray)
    assert dec.shape == DEFAULT_SHAPE
    assert dec.dtype == np.float32

    # out= parameter honored
    out = np.empty(DEFAULT_SHAPE, dtype=np.float32)
    ret = codec.decode(enc, out=out)
    assert ret is out
    np.testing.assert_array_equal(ret, dec)


def test_two_dimensional_challenge_slice_roundtrip():
    field = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    codec = HoapsWvpaCodec(shape=field.shape, error_bound=0.05)
    encoded = codec.encode(field)
    out = np.empty(field.shape, dtype=np.float32)
    decoded = codec.decode(encoded, out=out)
    assert decoded is out
    assert decoded.shape == field.shape
    assert np.isnan(decoded[0, 1])
    np.testing.assert_allclose(decoded[~np.isnan(field)], field[~np.isnan(field)], atol=0.05)


def test_decode_out_wrong_size_raises(codec_factory):
    codec = codec_factory()
    field, _ = make_smooth_field(shape=(2, 4, 4))
    codec2 = HoapsWvpaCodec(shape=(2, 4, 4), error_bound=0.05)
    enc = codec2.encode(field)
    bad = np.empty((2, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="out buffer"):
        codec2.decode(enc, out=bad)


def test_decode_rejects_unknown_scan_version(codec_factory, smooth_field):
    from hoaps_compressor.container import read_container, write_container

    field, _ = smooth_field
    codec = codec_factory()
    container = read_container(codec.encode(field))
    incompatible = write_container(
        header_extra=container.header_extra,
        mask_payload=container.mask_payload,
        residual_payload=container.residual_payload,
        model_version=999,
        outer_compress=False,
    )
    with pytest.raises(ValueError, match="model version"):
        codec.decode(incompatible)


def test_encode_wrong_buffer_size_raises(codec_factory):
    codec = codec_factory()
    with pytest.raises(ValueError, match="elements"):
        codec.encode(np.zeros(10, dtype=np.float32))


def test_constructor_validation():
    with pytest.raises(ValueError, match="error_bound"):
        HoapsWvpaCodec(shape=(1, 4, 4), error_bound=-0.1)
    with pytest.raises(ValueError, match="error_bound"):
        HoapsWvpaCodec(shape=(1, 4, 4), error_bound=float("inf"))
    with pytest.raises(ValueError, match="error_bound"):
        HoapsWvpaCodec(shape=(1, 4, 4), error_bound=float("nan"))
    # zero bound is VALID (FR-017)
    HoapsWvpaCodec(shape=(1, 4, 4), error_bound=0.0)
    with pytest.raises(ValueError, match="shape"):
        HoapsWvpaCodec(shape=(0, 4, 4), error_bound=0.1)
    with pytest.raises(ValueError, match="dtype"):
        HoapsWvpaCodec(shape=(1, 4, 4), error_bound=0.1, dtype="float64")
