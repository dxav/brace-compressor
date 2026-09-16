#!/usr/bin/env python3
"""Analyze the current entropy stage on a real HOAPS field.

Measures, per block: mode selection (RAW vs RANGE vs CTX), varint vs raw
sizes, rANS/CTX table overhead, and the unconditional + conditional
entropy of the symbol stream. This tells us where the entropy coder is
losing bits vs the entropy floor, and whether a context entropy
model could help.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hoaps_compressor import HoapsWvpaCodec
from hoaps_compressor.container import read_container
from hoaps_compressor.model.entropy import (
    BLOCK,
    MODE_CTX,
    MODE_RAW,
    MODE_RANGE,
    _HEAD,
    _RAWLEN,
    _TBL,
    decode_symbols,
    encode_symbols,
    varint_decode,
    varint_encode,
    zigzag,
)


def analyze(field: np.ndarray, bound: float) -> dict:
    codec = HoapsWvpaCodec(shape=field.shape, error_bound=bound)
    encoded = codec.encode(field)
    c = read_container(encoded)
    payload = c.residual_payload
    (ent_len,) = np.frombuffer(payload[:8], dtype="<u8")
    ent = payload[8 : 8 + ent_len]

    # Re-derive symbols by decoding (we don't have them directly here).
    n_valid = int(c.header_extra["n_valid"])
    symbols = decode_symbols(ent, n_valid)

    # --- Per-block mode analysis ---------------------------------------
    n_blocks = (n_valid + BLOCK - 1) // BLOCK
    off = _HEAD.size
    (nb,) = _HEAD.unpack_from(ent, 0)
    mode_bytes = ent[off : off + nb]
    off += nb
    (raw_len,) = _RAWLEN.unpack_from(ent, off)
    off += _RAWLEN.size
    raw_section = ent[off : off + raw_len]
    off += raw_len

    n_raw = 0
    n_range = 0
    n_ctx = 0
    raw_bytes = 0
    range_varint_bytes = 0
    range_sym_count = 0
    for bi in range(n_blocks):
        mbyte = mode_bytes[bi]
        width, mode = mbyte >> 4, mbyte & 0x0F
        count = min(BLOCK, n_valid - bi * BLOCK)
        if mode == MODE_RAW:
            n_raw += 1
            raw_bytes += width * count
        elif mode == MODE_CTX:
            n_ctx += 1
        else:
            n_range += 1
            range_varint_bytes += len(varint_encode(zigzag(symbols[bi * BLOCK : bi * BLOCK + count])))
            range_sym_count += count

    # rANS section overhead
    rans_overhead = 0
    if n_range:
        counts_rb = np.frombuffer(ent[off : off + 2 * n_range], dtype="<u2")
        off += 2 * n_range
        (total_syms,) = np.frombuffer(ent[off : off + 4], dtype="<u4")
        off += 4
        (tbl_len,) = _TBL.unpack_from(ent, off)
        off += _TBL.size
        table_varints = ent[off : off + tbl_len]
        off += tbl_len
        freqs = varint_decode(table_varints, 256).astype(np.int64)
        (rans_len,) = _TBL.unpack_from(ent, off)
        off += _TBL.size
        stream = ent[off : off + rans_len]
        rans_overhead = 2 * n_range + 4 + _TBL.size + tbl_len + _TBL.size

    # CTX section overhead
    ctx_overhead = 0
    if n_ctx:
        (span,) = struct.unpack_from("<I", ent, off)
        off += 4
        (min_symbol,) = struct.unpack_from("<q", ent, off)
        off += 8
        (n_ctx_tables,) = struct.unpack_from("<I", ent, off)
        off += 4
        for _ in range(n_ctx_tables):
            (tbl_len,) = _TBL.unpack_from(ent, off)
            off += _TBL.size
            off += tbl_len
        (rans_len,) = _TBL.unpack_from(ent, off)
        off += _TBL.size
        ctx_overhead = 4 + 8 + 4 + n_ctx_tables * _TBL.size + rans_len + _TBL.size
        ctx_stream_bytes = rans_len

    # --- Symbol entropy (empirical) ------------------------------------
    counts = np.bincount(zigzag(symbols), minlength=0).astype(np.float64)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    entropy_bits = float(-(p * np.log2(p)).sum())

    # --- Conditional entropy (first-order, by previous magnitude bucket) -
    cond_bits = 0.0
    if n_valid > 1:
        prev = symbols[:-1]
        cur = symbols[1:]
        a = np.abs(prev)
        buckets = np.where(a <= 1, 0, np.where(a <= 8, 1, 2))
        for b in range(3):
            sel = buckets == b
            if sel.sum() == 0:
                continue
            sub = cur[sel]
            c = np.bincount(sub - sub.min(), minlength=0).astype(np.float64)
            c = c[c > 0]
            pb = c / c.sum()
            cond_bits += (sel.sum() / cur.size) * float(-(pb * np.log2(pb)).sum())

    # --- What a perfect entropy coder would need ------------------------
    # Ideal: entropy_bits per symbol, no table/block overhead.
    ideal_bytes = entropy_bits * n_valid / 8.0
    ideal_cond_bytes = cond_bits * n_valid / 8.0

    return {
        "n_valid": n_valid,
        "n_blocks": n_blocks,
        "n_raw": n_raw,
        "n_range": n_range,
        "n_ctx": n_ctx,
        "raw_bytes": raw_bytes,
        "range_varint_bytes": range_varint_bytes,
        "range_sym_count": range_sym_count,
        "rans_overhead": rans_overhead,
        "ctx_overhead": ctx_overhead,
        "ctx_stream_bytes": ctx_stream_bytes,
        "entropy_bits_per_symbol": entropy_bits,
        "cond_entropy_bits_per_symbol": cond_bits,
        "ideal_bytes": ideal_bytes,
        "ideal_cond_bytes": ideal_cond_bytes,
        "actual_entropy_bytes": len(ent),
        "total_payload_bytes": len(payload),
        "compressed_size": len(encoded),
        "cr": field.nbytes / len(encoded),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/wvpa_2020-08-01_07.npy")
    ap.add_argument("--bound", type=float, default=0.05)
    args = ap.parse_args()

    field = np.load(args.input).astype(np.float32)
    if field.ndim == 2:
        field = field[None, ...]
    field = np.ascontiguousarray(field)

    r = analyze(field, args.bound)
    print(f"field {field.shape}  bound {args.bound}")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:12.4f}")
        else:
            print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()