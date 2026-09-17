"""BRCE (BRACE Container Encoding) stream framing.

Layout (all little-endian):

| Offset | Size | Field                                        |
|--------|------|----------------------------------------------|
| 0      | 4    | Magic b"BRCE"                                |
| 4      | 2    | Container version (uint16, ABI 3)            |
| 6      | 2    | Flags (Zstandard payload compression selectors) |
| 8      | 4    | Model version (uint32)                       |
| 12     | 4    | Header-extra JSON length H (uint32)          |
| 16     | H    | Header extra (UTF-8 JSON)                    |
| 16+H   | 8    | Mask payload length (uint64)                 |
| ...    | var  | Mask payload (RLE or bitpacked, lossless)    |
| ...    | 8    | Residual payload length (uint64)             |
| ...    | var  | Residual payload                             |
| ...    | 4    | CRC-32 over all preceding bytes              |
"""

from __future__ import annotations

import json
import struct
import binascii
from dataclasses import dataclass

import zstandard

MAGIC = b"BRCE"
CONTAINER_VERSION = 3

# Flag 0x0001 means both payloads are Zstandard-compressed.
# Flags 0x0002/0x0004 identify mask-only or residual-only compression.
FLAG_OUTER_COMPRESSED = 0x0001
FLAG_MASK_COMPRESSED = 0x0002
FLAG_RESIDUAL_COMPRESSED = 0x0004

_HEADER_PREFIX = struct.Struct("<4sHHII")
_LEN64 = struct.Struct("<Q")
_CRC32 = struct.Struct("<I")


class ContainerError(ValueError):
    """Raised when container framing, version or checksum validation fails."""


@dataclass
class Container:
    """Decoded representation of an encoded stream."""

    container_version: int
    flags: int
    model_version: int
    header_extra: dict
    mask_payload: bytes
    residual_payload: bytes

    @property
    def outer_compressed(self) -> bool:
        return bool(self.flags & (FLAG_OUTER_COMPRESSED | FLAG_MASK_COMPRESSED | FLAG_RESIDUAL_COMPRESSED))

    @property
    def mask_outer_compressed(self) -> bool:
        return bool(self.flags & (FLAG_OUTER_COMPRESSED | FLAG_MASK_COMPRESSED))

    @property
    def residual_outer_compressed(self) -> bool:
        return bool(self.flags & (FLAG_OUTER_COMPRESSED | FLAG_RESIDUAL_COMPRESSED))


def _compress_if_needed(data: bytes, compress: bool) -> bytes:
    if compress and data:
        return zstandard.ZstdCompressor(level=3).compress(data)
    return data


def _decompress_if_needed(data: bytes, compressed: bool) -> bytes:
    if compressed and data:
        try:
            return zstandard.ZstdDecompressor().decompress(data)
        except zstandard.ZstdError as exc:
            raise ContainerError(f"payload decompression failed: {exc}") from exc
    return data


def write_container(
    header_extra: dict,
    mask_payload: bytes,
    residual_payload: bytes,
    model_version: int,
    outer_compress: bool = False,
) -> bytes:
    """Frame header + payloads + CRC-32 into the encoded stream bytes."""
    header_json = json.dumps(header_extra, separators=(",", ":"), sort_keys=True)
    header_bytes = header_json.encode("utf-8")

    mask_out, residual_out = mask_payload, residual_payload
    flags = 0
    if outer_compress:
        # Outer compression is a CR-maximizing, always-lossless pass. Select
        # each payload independently because entropy-coded residuals often do
        # not shrink while the bitpacked missing mask does.
        mask_c = _compress_if_needed(mask_payload, True)
        res_c = _compress_if_needed(residual_payload, True)
        mask_better = (not mask_payload) or len(mask_c) < len(mask_payload)
        res_better = (not residual_payload) or len(res_c) < len(residual_payload)
        if mask_better:
            mask_out = mask_c
            flags |= FLAG_MASK_COMPRESSED
        if res_better:
            residual_out = res_c
            flags |= FLAG_RESIDUAL_COMPRESSED
        if mask_better and res_better:
            # Preserve the original ABI-1 flag for streams where both payloads
            # are compressed.
            flags = FLAG_OUTER_COMPRESSED

    parts = [
        _HEADER_PREFIX.pack(
            MAGIC, CONTAINER_VERSION, flags, model_version, len(header_bytes)
        ),
        header_bytes,
        _LEN64.pack(len(mask_out)),
        mask_out,
        _LEN64.pack(len(residual_out)),
        residual_out,
    ]
    body = b"".join(parts)
    return body + _CRC32.pack(binascii.crc32(body) & 0xFFFFFFFF)


def read_container(buf) -> Container:
    """Parse and validate an encoded stream. Raises ContainerError(ValueError)."""
    data = bytes(buf)
    if len(data) >= _HEADER_PREFIX.size:
        # Structural checks first (nicer errors than a bare CRC mismatch).
        magic, version = _HEADER_PREFIX.unpack(data[: _HEADER_PREFIX.size])[:2]
        if magic != MAGIC:
            raise ContainerError(f"bad magic {magic!r}; not a BRCE container")
        if version != CONTAINER_VERSION:
            raise ContainerError(
                f"unsupported container version {version}; "
                f"this codec supports {CONTAINER_VERSION}"
            )

    if len(data) < _HEADER_PREFIX.size + _CRC32.size:
        raise ContainerError(
            f"encoded stream too short: {len(data)} bytes; not a BRCE container"
        )

    crc_stored = _CRC32.unpack(data[-_CRC32.size:])[0]
    body = data[:-_CRC32.size]
    crc_actual = binascii.crc32(body) & 0xFFFFFFFF
    if crc_stored != crc_actual:
        raise ContainerError(
            f"container checksum mismatch: stored {crc_stored:#x}, computed {crc_actual:#x}"
        )

    magic, version, flags, model_version, header_len = _HEADER_PREFIX.unpack(
        body[: _HEADER_PREFIX.size]
    )

    offset = _HEADER_PREFIX.size
    header_bytes = body[offset : offset + header_len]
    if len(header_bytes) != header_len:
        raise ContainerError("truncated header extra")
    try:
        header_extra = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContainerError(f"invalid header-extra JSON: {exc}") from exc
    offset += header_len

    def _take_payload() -> bytes:
        nonlocal offset
        if offset + _LEN64.size > len(body):
            raise ContainerError("truncated payload length field")
        (plen,) = _LEN64.unpack(body[offset : offset + _LEN64.size])
        offset += _LEN64.size
        payload = body[offset : offset + plen]
        if len(payload) != plen:
            raise ContainerError("truncated payload section")
        offset += plen
        return payload

    mask_payload = _take_payload()
    residual_payload = _take_payload()

    mask_payload = _decompress_if_needed(
        mask_payload,
        bool(flags & (FLAG_OUTER_COMPRESSED | FLAG_MASK_COMPRESSED)),
    )
    residual_payload = _decompress_if_needed(
        residual_payload,
        bool(flags & (FLAG_OUTER_COMPRESSED | FLAG_RESIDUAL_COMPRESSED)),
    )

    return Container(
        container_version=version,
        flags=flags,
        model_version=model_version,
        header_extra=header_extra,
        mask_payload=mask_payload,
        residual_payload=residual_payload,
    )
