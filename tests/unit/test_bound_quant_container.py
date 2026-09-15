"""Unit tests: bound validation and quantization (T031) and container (T032)."""

import numpy as np
import pytest

from hoaps_compressor.bound import normalize_missing_value, validate_error_bound, validate_shape
from hoaps_compressor.container import (
    ContainerError,
    read_container,
    write_container,
)
from hoaps_compressor.quant import (
    dequantize,
    derive_step,
    quantization_error_bound,
    quantize,
)


class TestErrorBound:
    def test_valid_bounds(self):
        assert validate_error_bound(0.0) == 0.0  # FR-017: zero valid
        assert validate_error_bound(0.05) == 0.05
        assert validate_error_bound(1e6) == 1e6

    @pytest.mark.parametrize("bad", [-0.1, float("inf"), float("-inf"), float("nan")])
    def test_invalid_bounds(self, bad):
        with pytest.raises(ValueError):
            validate_error_bound(bad)

    def test_non_numeric(self):
        with pytest.raises(TypeError):
            validate_error_bound("0.1")  # type: ignore[arg-type]

    def test_shape_validation(self):
        assert validate_shape((2, 3, 4)) == (2, 3, 4)
        for bad in [(0, 3, 4), (-1, 3, 4), (2, 3), (2, 3, 4, 5), (1.5, 3, 4)]:
            with pytest.raises(ValueError):
                validate_shape(bad)

    def test_missing_value_normalization(self):
        assert np.isnan(normalize_missing_value("nan"))
        assert normalize_missing_value(-999.0) == -999.0
        with pytest.raises(ValueError):
            normalize_missing_value("fill")
        with pytest.raises((ValueError, TypeError)):
            normalize_missing_value(float("inf"))


class TestQuantization:
    def test_derive_step(self):
        assert derive_step(0.1) == pytest.approx(0.2)  # Δ = 2·bound (full budget)
        assert derive_step(0.0) == 0.0  # bound 0 -> exact mode
        with pytest.raises(ValueError):
            derive_step(-1.0)

    def test_quantization_error_bounded(self):
        step = derive_step(0.1)
        residual = np.linspace(-50, 50, 1001)
        syms = quantize(residual, step)
        rec = dequantize(syms, step)
        assert np.abs(rec - residual).max() <= quantization_error_bound(step) + 1e-12

    def test_roundtrip_random(self):
        rng = np.random.default_rng(3)
        step = derive_step(0.2)
        residual = rng.normal(0, 10, 10_000)
        syms = quantize(residual, step, origin=float(residual.mean()))
        rec = dequantize(syms, step, origin=float(residual.mean()))
        assert np.abs(rec - residual).max() <= step / 2 + 1e-12


class TestContainer:
    def test_write_read_roundtrip(self):
        hdr = {"a": 1, "b": [1.5, 2.5], "s": "x"}
        mask = b"\xf0" * 31
        res = b"\x01\x02\x03"
        data = write_container(hdr, mask, res, model_version=2)
        c = read_container(data)
        assert c.header_extra == hdr
        assert c.mask_payload == mask
        assert c.residual_payload == res
        assert c.model_version == 2

    def test_outer_compression_roundtrip(self):
        hdr = {"k": "v"}
        mask = bytes(1000)
        res = bytes(2000)
        data = write_container(hdr, mask, res, model_version=1, outer_compress=True)
        c = read_container(data)
        assert c.outer_compressed
        assert c.mask_payload == mask
        assert c.residual_payload == res
        assert len(data) < len(write_container(hdr, mask, res, 1, False))

    def test_bad_magic(self):
        data = bytearray(write_container({}, b"", b"", 1))
        data[0:4] = b"XXXX"
        with pytest.raises(ContainerError, match="magic"):
            read_container(bytes(data))

    def test_checksum_mismatch(self):
        data = bytearray(write_container({}, b"\x01", b"\x02", 1))
        data[-1] ^= 0xFF
        with pytest.raises(ContainerError, match="checksum"):
            read_container(bytes(data))

    def test_bad_version(self):
        data = bytearray(write_container({}, b"", b"", 1))
        data[4:6] = (99).to_bytes(2, "little")
        with pytest.raises(ContainerError, match="version"):
            read_container(bytes(data))

    def test_truncated(self):
        with pytest.raises(ContainerError):
            read_container(b"HW")


class TestMaskUnit:
    def test_pack_unpack_bitexact(self):
        from hoaps_compressor.mask import pack_mask, unpack_mask

        rng = np.random.default_rng(5)
        mask = rng.random(1000) > 0.5
        packed = pack_mask(mask)
        assert len(packed) == (1000 + 7) // 8
        np.testing.assert_array_equal(unpack_mask(packed, 1000), mask)

    def test_extract_mask_nan_and_finite(self):
        from hoaps_compressor.mask import extract_mask

        vals = np.array([1.0, np.nan, 2.0, np.inf, -999.0], dtype=np.float32)
        m = extract_mask(vals, "nan")
        np.testing.assert_array_equal(m, [False, True, False, True, False])
        m2 = extract_mask(vals, -999.0)
        np.testing.assert_array_equal(m2, [False, True, False, True, True])
