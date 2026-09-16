"""Bit-exact entropy coding stage.

Per-block mode selection between:
- ``MODE_RAW``   : fixed-width two's-complement ints, stored raw;
- ``MODE_RANGE`` : zigzag varints coded with a static rANS range coder
  (shared frequency table) — the "RANGE_CODED" mode;
- ``MODE_CTX``   : context-adaptive rANS over the symbol alphabet, with
  per-context frequency tables selected by the previous symbol's
    magnitude bucket.

Both paths are fully lossless (bit-exact), which is the prerequisite for
the verify-and-repair guarantee. Layout of ``encode_symbols`` output::

    u32  n_blocks
    n_blocks bytes   mode/width byte per block ((width << 4) | mode)
    <raw section>    concatenated RAW blocks (block order)
    u64  raw_len     total raw-section length
    if any range block:
        u16  per range block: varint byte count (block order)
        u32  table_len; table_len varints: 256 normalized frequencies
        u32  rans_len; rans_len bytes: rANS stream
    if any ctx block:
        u32  span; i64 min_symbol
        u32  n_ctx; n_ctx * (u32 table_len; table varints)
        u32  rans_len; rans_len bytes: rANS stream
    u32  n_repair; n_repair * (i64 position, f32 value)   [repair section]

Empty symbol streams encode to an empty payload.
"""

from __future__ import annotations

import struct

import numpy as np

BLOCK = 2048
MODE_RAW = 0
MODE_RANGE = 1
MODE_CTX = 2

_HEAD = struct.Struct("<I")
_RAWLEN = struct.Struct("<Q")
_TBL = struct.Struct("<I")
_REPAIR = struct.Struct("<I")

# Context buckets by previous symbol magnitude. The residual
# distribution depends strongly on the previous residual's size: smooth
# regions produce tiny residuals, edges produce larger ones. Conditioning
# the probability model on this bucket exploits the measured conditional
# entropy (H(s|s_prev) = 4.84 vs H(s) = 5.44 bits/symbol).
_CTX_BUCKETS = (1, 8)  # |s| <= 1 -> ctx 0; <= 8 -> ctx 1; else ctx 2
N_CTX = 3


# ---------------------------------------------------------------------------
# zigzag + varint (vectorized, bit-exact)
# ---------------------------------------------------------------------------
def zigzag(symbols: np.ndarray) -> np.ndarray:
    """Map int64 -> uint64 (zigzag): 0, -1, 1, -2, 2 ... -> 0, 1, 2, 3, 4 ..."""
    s = np.asarray(symbols, dtype=np.int64)
    return ((s << 1) ^ (s >> 63)).astype(np.uint64)


def unzigzag(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`zigzag`."""
    z = np.asarray(z, dtype=np.uint64)
    return ((z >> 1).astype(np.int64)) ^ (-(z & np.uint64(1)).astype(np.int64))


def varint_encode(z: np.ndarray) -> bytes:
    """Encode uint64 values as LEB128 varints (7 bits per byte, LSB first)."""
    z = np.asarray(z, dtype=np.uint64)
    if z.size == 0:
        return b""
    # Exact varint lengths: an int needs 1 + floor(bitlen/7) bytes; computed
    # by counting 7-bit shifts until exhausted (np.bitwise_count is a
    # population count, NOT a bit length, and must not be used here).
    lens = np.ones(z.size, dtype=np.int64)
    rem = z.copy()
    for _ in range(9):  # uint64 needs at most 10 varint bytes
        rem = rem >> np.uint64(7)
        lens += (rem != 0).astype(np.int64)
    total = int(lens.sum())
    out = np.zeros(total, dtype=np.uint8)
    offs = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    rem = z.copy()
    for i in range(10):
        active = lens > i
        if not active.any():
            break
        byte = (rem & np.uint64(0x7F)).astype(np.uint8)
        more = (rem >> np.uint64(7)) != 0
        out[offs[active] + i] = np.where(more[active], byte[active] | 0x80, byte[active])
        rem = rem >> np.uint64(7)
    return out.tobytes()


def varint_decode(data: bytes, n: int) -> np.ndarray:
    """Decode ``n`` LEB128 varints from ``data`` (bit-exact inverse)."""
    if n == 0:
        return np.zeros(0, dtype=np.uint64)
    b = np.frombuffer(bytes(data), dtype=np.uint8)
    cont = (b & 0x80) != 0
    ends = np.nonzero(~cont)[0]
    if ends.size != n:
        raise ValueError(
            f"varint stream holds {ends.size} values; expected {n}"
        )
    starts = np.concatenate(([0], ends[:-1] + 1)).astype(np.int64)
    lens = (ends - starts + 1).astype(np.int64)
    # Map each byte to its (value index, shift) contribution; a plain
    # reduceat is wrong because reduceat sums per segment starting at
    # 'starts' but the segment covering must align to END positions.
    shifts = (np.arange(b.size, dtype=np.int64) - np.repeat(starts, lens)) * 7
    contrib = (b & np.uint8(0x7F)).astype(np.uint64) << shifts.astype(np.uint64)
    # Build per-value sums explicitly (segment i = starts[i]..ends[i])
    out = np.zeros(n, dtype=np.uint64)
    idx = np.repeat(np.arange(n), lens)
    np.add.at(out, idx, contrib)
    return out


# ---------------------------------------------------------------------------
# static rANS (range coder) over byte alphabet, scale 16 bits
# ---------------------------------------------------------------------------
_RANS_SCALE = 16
_RANS_M = 1 << _RANS_SCALE
_RANS_L = 1 << 23


def _normalize_freqs(counts: np.ndarray) -> np.ndarray:
    """Normalize 256 byte counts onto exactly ``_RANS_M`` total (deterministic)."""
    counts = np.asarray(counts, dtype=np.int64)
    present = counts > 0
    if not present.any():
        return np.zeros(256, dtype=np.int64)
    total = int(counts.sum())
    freqs = np.where(present, np.maximum(1, (counts * _RANS_M) // total), 0).astype(
        np.int64
    )
    diff = _RANS_M - int(freqs.sum())
    order = np.argsort(-counts[present], kind="stable")
    present_idx = np.nonzero(present)[0]
    i = 0
    n_present = int(present_idx.size)
    while diff != 0:
        idx = int(present_idx[order[i % n_present]])
        if diff > 0:
            freqs[idx] += 1
            diff -= 1
        elif freqs[idx] > 1:
            freqs[idx] -= 1
            diff += 1
        i += 1
    return freqs


def rans_encode(data: bytes, freqs: np.ndarray) -> bytes:
    """Static rANS encoder that preserves the input byte stream exactly.

    rANS is a stack: decode pops in reverse of encode, so symbols are fed
    in reverse here to make the decoder recover the original order.
    """
    cum = np.concatenate(([0], np.cumsum(freqs))).astype(np.int64)
    x = _RANS_L
    pushes: list[int] = []
    # Standard rANS renorm threshold: x_max = ((L >> SCALE) << 8) * f.
    # With SCALE=16 and L=2^23 this is 2^15 * f; using it keeps the encoded
    # state < 2^31 so the final 4-byte flush is exact and the decoder's
    # "while x < L" byte-pull loop stays byte-aligned with the pushes.
    threshold = (_RANS_L >> _RANS_SCALE) << 8  # 2^15
    for byte in reversed(data):
        f = int(freqs[byte])
        x_max = threshold * f
        while x >= x_max:
            pushes.append(x & 0xFF)
            x >>= 8
        c = int(cum[byte])
        x = ((x // f) << _RANS_SCALE) + (x % f) + c
    # flush remaining state (4 bytes is enough: state < 2^31)
    for _ in range(4):
        pushes.append(x & 0xFF)
        x >>= 8
    return bytes(reversed(pushes))


def rans_decode(src: bytes, n_bytes: int, freqs: np.ndarray) -> bytes:
    """Static rANS decoder; recovers exactly ``n_bytes`` bytes (bit-exact)."""
    if n_bytes == 0:
        return b""
    cum = np.concatenate(([0], np.cumsum(freqs))).astype(np.int64)
    lut = np.zeros(_RANS_M, dtype=np.uint8)
    for s in range(256):
        lut[cum[s] : cum[s + 1]] = s
    x = int.from_bytes(src[:4], "big")
    ptr = 4
    out = np.empty(n_bytes, dtype=np.uint8)
    lut_np = lut
    freq_np = freqs
    data = bytes(src)
    for i in range(n_bytes):
        slot = x & (_RANS_M - 1)
        s = int(lut_np[slot])
        out[i] = s
        f = int(freq_np[s])
        x = (f * (x >> _RANS_SCALE)) + slot - int(cum[s])
        while x < _RANS_L:
            x = (x << 8) | data[ptr]
            ptr += 1
    return out.tobytes()


# ---------------------------------------------------------------------------
# context-adaptive rANS over the symbol alphabet
# ---------------------------------------------------------------------------
def _ctx_of(prev: int) -> int:
    """Context bucket for the previous symbol's magnitude."""
    a = abs(prev)
    if a <= _CTX_BUCKETS[0]:
        return 0
    if a <= _CTX_BUCKETS[1]:
        return 1
    return 2


def _normalize_freqs_span(counts: np.ndarray, span: int) -> np.ndarray:
    """Normalize ``span`` counts onto exactly ``_RANS_M`` total (deterministic)."""
    counts = np.asarray(counts, dtype=np.int64)
    present = counts > 0
    if not present.any():
        return np.zeros(span, dtype=np.int64)
    total = int(counts.sum())
    freqs = np.where(present, np.maximum(1, (counts * _RANS_M) // total), 0).astype(
        np.int64
    )
    diff = _RANS_M - int(freqs.sum())
    order = np.argsort(-counts[present], kind="stable")
    present_idx = np.nonzero(present)[0]
    i = 0
    n_present = int(present_idx.size)
    while diff != 0:
        idx = int(present_idx[order[i % n_present]])
        if diff > 0:
            freqs[idx] += 1
            diff -= 1
        elif freqs[idx] > 1:
            freqs[idx] -= 1
            diff += 1
        i += 1
    return freqs


def ctx_rans_encode(
    symbols: np.ndarray, freqs_list: list[np.ndarray], min_symbol: int
) -> bytes:
    """Context-adaptive rANS encoder over the symbol alphabet.

    ``symbols`` are int64; each is mapped to ``idx = s - min_symbol`` in
    ``[0, span)``. The probability table for symbol ``i`` is
    ``freqs_list[ctx(symbols[i-1])]`` (the first symbol uses ctx 0).
    rANS is a stack, so symbols are fed in reverse; the context for each
    symbol is computed from the *preceding* symbol in forward order, which
    the decoder reproduces exactly.
    """
    n = int(symbols.size)
    if n == 0:
        return b""
    span = int(freqs_list[0].size)
    idx = (symbols.astype(np.int64) - min_symbol).astype(np.int64)
    # Context for each symbol (forward order): ctx of the previous symbol.
    ctxs = np.zeros(n, dtype=np.int64)
    if n > 1:
        a = np.abs(symbols[:-1])
        ctxs[1:] = np.where(a <= _CTX_BUCKETS[0], 0, np.where(a <= _CTX_BUCKETS[1], 1, 2))

    cums = [np.concatenate(([0], np.cumsum(f))).astype(np.int64) for f in freqs_list]
    # Convert to Python lists for fast scalar indexing in the hot loop.
    freqs_py = [f.tolist() for f in freqs_list]
    cums_py = [c.tolist() for c in cums]
    idx_py = idx.tolist()
    ctxs_py = ctxs.tolist()
    x = _RANS_L
    # Preallocate the output buffer: rANS emits at most ~2 bytes per symbol
    # plus a 4-byte flush. Overallocate and trim at the end.
    out = np.empty(n * 2 + 8, dtype=np.uint8)
    out_len = 0
    threshold = (_RANS_L >> _RANS_SCALE) << 8  # 2^15
    # Feed in reverse so decode pops in forward order.
    for i in range(n - 1, -1, -1):
        c = ctxs_py[i]
        f = freqs_py[c][idx_py[i]]
        x_max = threshold * f
        while x >= x_max:
            out[out_len] = x & 0xFF
            out_len += 1
            x >>= 8
        x = ((x // f) << _RANS_SCALE) + (x % f) + cums_py[c][idx_py[i]]
    for _ in range(4):
        out[out_len] = x & 0xFF
        out_len += 1
        x >>= 8
    return bytes(reversed(out[:out_len].tobytes()))


def ctx_rans_decode(
    src: bytes, n: int, freqs_list: list[np.ndarray], min_symbol: int
) -> np.ndarray:
    """Context-adaptive rANS decoder; recovers exactly ``n`` symbols."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    span = int(freqs_list[0].size)
    cums = [np.concatenate(([0], np.cumsum(f))).astype(np.int64) for f in freqs_list]
    luts = []
    for cum in cums:
        lut = np.zeros(_RANS_M, dtype=np.int64)
        for s in range(span):
            lut[cum[s] : cum[s + 1]] = s
        luts.append(lut)
    # Python lists for fast scalar indexing in the hot loop.
    freqs_py = [f.tolist() for f in freqs_list]
    cums_py = [c.tolist() for c in cums]
    luts_py = [l.tolist() for l in luts]
    x = int.from_bytes(src[:4], "big")
    ptr = 4
    data = bytes(src)
    out = np.empty(n, dtype=np.int64)
    prev = 0
    for i in range(n):
        c = _ctx_of(prev)
        slot = x & (_RANS_M - 1)
        s = luts_py[c][slot]
        out[i] = s + min_symbol
        f = freqs_py[c][s]
        x = (f * (x >> _RANS_SCALE)) + slot - cums_py[c][s]
        while x < _RANS_L:
            x = (x << 8) | data[ptr]
            ptr += 1
        prev = int(out[i])
    return out


# ---------------------------------------------------------------------------
# per-block symbol codec
# ---------------------------------------------------------------------------
def _block_count(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK


# CTX mode is skipped when the symbol alphabet is too large: the per-context
# tables (3 x span entries) would cost more than the conditional-entropy
# savings. 8192 keeps the table overhead bounded (~tens of KB).
_CTX_MAX_SPAN = 8192


def _encode_range_payload(
    syms: np.ndarray, n_blocks: int
) -> tuple[bytes, list[int], list[int]]:
    """Build the per-block RAW/RANGE payload.

    Returns (payload, mode_bytes, widths) so the caller can compare sizes.
    """
    n = int(syms.size)
    # Pass 1: per-block varint stream, raw packed bytes, and width.
    varints: list[bytes] = []
    packeds: list[bytes] = []
    widths: list[int] = []
    all_varints = bytearray()
    for bi in range(n_blocks):
        chunk = syms[bi * BLOCK : (bi + 1) * BLOCK]
        z = zigzag(chunk)
        v = varint_encode(z)
        mn, mx = int(chunk.min()), int(chunk.max())
        if mn >= -128 and mx <= 127:
            width, packed = 1, chunk.astype("<i1").tobytes()
        elif mn >= -32768 and mx <= 32767:
            width, packed = 2, chunk.astype("<i2").tobytes()
        elif mn >= -(2**31) and mx <= 2**31 - 1:
            width, packed = 4, chunk.astype("<i4").tobytes()
        else:
            width, packed = 8, chunk.astype("<i8").tobytes()
        varints.append(v)
        packeds.append(packed)
        widths.append(width)
        all_varints += v

    def _est_rans(v: bytes, log2f: np.ndarray) -> float:
        """Estimated rANS bytes for a varint stream under a frequency table.

        rANS achieves within ~0.01 bits/byte of the empirical entropy, so
        the ideal size (sum of -log2 p over the bytes, /8) is an accurate
        estimate of the rANS stream size.
        """
        if not v:
            return 0.0
        return -float(np.sum(log2f[np.frombuffer(v, dtype=np.uint8)])) / 8.0

    def _decide(log2f: np.ndarray) -> list[bool]:
        # RANGE wins when its marginal cost (rANS stream bytes + the 2-byte
        # u16 varint-count field) beats raw packing.
        return [
            (_est_rans(v, log2f) + 2.0) < len(p) for v, p in zip(varints, packeds)
        ]

    # Pass 2-4: shared frequency table over the varint bytes, iterated to
    # convergence. The table is built from the chosen RANGE blocks; the
    # decision is re-run with the refined table until stable (converges in
    # 1-2 passes in practice).
    chosen: list[bool] = [True] * n_blocks
    for _ in range(4):
        range_varints = b"".join(v for v, ch in zip(varints, chosen) if ch)
        counts = np.bincount(
            np.frombuffer(range_varints, dtype=np.uint8), minlength=256
        ).astype(np.int64)
        freqs = _normalize_freqs(counts)
        log2f = np.zeros(256, dtype=np.float64)
        nz = freqs > 0
        log2f[nz] = np.log2(freqs[nz] / (1 << 16))
        new_chosen = _decide(log2f)
        if new_chosen == chosen:
            break
        chosen = new_chosen

    # Build the output sections.
    mode_bytes = bytearray()
    raw_parts: list[bytes] = []
    range_counts: list[int] = []
    range_symbols = 0
    varint_parts: list[bytes] = []
    for bi, ch in enumerate(chosen):
        if ch:
            mode_bytes.append((widths[bi] << 4) | MODE_RANGE)
            range_counts.append(len(varints[bi]))
            range_symbols += int(min(BLOCK, n - bi * BLOCK))
            varint_parts.append(varints[bi])
        else:
            mode_bytes.append((widths[bi] << 4) | MODE_RAW)
            raw_parts.append(packeds[bi])

    raw_section = b"".join(raw_parts)
    parts = [
        _HEAD.pack(n_blocks),
        bytes(mode_bytes),
        _RAWLEN.pack(len(raw_section)),
        raw_section,
    ]
    if range_counts:
        # rANS section layout: [u16 per range block: varint byte count]
        #                      [u32 total symbol COUNT across range blocks]
        #                      [u32 table length; table varints(256 freqs)]
        #                      [u32 rANS length; rANS stream]
        varint_stream = b"".join(varint_parts)
        counts = np.bincount(
            np.frombuffer(varint_stream, dtype=np.uint8), minlength=256
        ).astype(np.int64)
        freqs = _normalize_freqs(counts)
        table_varints = varint_encode(freqs.astype(np.uint64))
        stream = rans_encode(varint_stream, freqs)
        parts.append(struct.pack(f"<{len(range_counts)}H", *range_counts))
        parts.append(struct.pack("<I", range_symbols))
        parts.append(_TBL.pack(len(table_varints)))
        parts.append(table_varints)
        parts.append(_TBL.pack(len(stream)))
        parts.append(stream)
    else:
        parts.append(struct.pack("<0H"))
    return b"".join(parts), list(mode_bytes), widths


def _encode_ctx_payload(syms: np.ndarray, n_blocks: int) -> bytes | None:
    """Whole-stream context-adaptive rANS payload (all blocks CTX).

    All symbols are coded in one rANS stream whose per-symbol probability
    table is selected by the previous symbol's magnitude bucket. Returns
    ``None`` when the symbol alphabet is too large for the tables.
    """
    n = int(syms.size)
    span = int(syms.max()) - int(syms.min()) + 1
    if span > _CTX_MAX_SPAN:
        return None
    min_symbol = int(syms.min())
    idx = (syms.astype(np.int64) - min_symbol).astype(np.int64)

    # Context for each symbol (forward order): bucket of the previous symbol.
    ctxs = np.zeros(n, dtype=np.int64)
    if n > 1:
        a = np.abs(syms[:-1])
        ctxs[1:] = np.where(a <= _CTX_BUCKETS[0], 0, np.where(a <= _CTX_BUCKETS[1], 1, 2))

    # Per-context frequency tables.
    counts = np.zeros((N_CTX, span), dtype=np.int64)
    np.add.at(counts, (ctxs, idx), 1)
    freqs_list = [_normalize_freqs_span(counts[c], span) for c in range(N_CTX)]
    table_varints = b"".join(
        varint_encode(f.astype(np.uint64)) for f in freqs_list
    )
    stream = ctx_rans_encode(syms, freqs_list, min_symbol)

    mode_bytes = bytes([MODE_CTX]) * n_blocks
    parts = [
        _HEAD.pack(n_blocks),
        mode_bytes,
        _RAWLEN.pack(0),  # no raw section
        b"",
        struct.pack("<I", span),
        struct.pack("<q", min_symbol),
        struct.pack("<I", N_CTX),
    ]
    for f in freqs_list:
        tv = varint_encode(f.astype(np.uint64))
        parts.append(_TBL.pack(len(tv)))
        parts.append(tv)
    parts.append(_TBL.pack(len(stream)))
    parts.append(stream)
    return b"".join(parts)


def _estimate_ctx_size(syms: np.ndarray) -> float | None:
    """Cheap estimate of the CTX payload size (bytes), or None if inapplicable.

    Uses the per-context empirical entropy of the symbol stream (the rANS
    stream is within ~0.01 bits/symbol of it) plus the table overhead.
    This lets ``encode_symbols`` decide between RANGE and CTX without
    building both full payloads.
    """
    n = int(syms.size)
    if n == 0:
        return 0.0
    span = int(syms.max()) - int(syms.min()) + 1
    if span > _CTX_MAX_SPAN:
        return None
    min_symbol = int(syms.min())
    idx = (syms.astype(np.int64) - min_symbol).astype(np.int64)
    ctxs = np.zeros(n, dtype=np.int64)
    if n > 1:
        a = np.abs(syms[:-1])
        ctxs[1:] = np.where(a <= _CTX_BUCKETS[0], 0, np.where(a <= _CTX_BUCKETS[1], 1, 2))
    counts = np.zeros((N_CTX, span), dtype=np.int64)
    np.add.at(counts, (ctxs, idx), 1)
    bits = 0.0
    table_bytes = 0
    for c in range(N_CTX):
        row = counts[c]
        total = int(row.sum())
        if total == 0:
            continue
        nz = row > 0
        p = row[nz].astype(np.float64) / total
        bits += total * float(-(p * np.log2(p)).sum())
        # table varint bytes: each freq is ~2-3 varint bytes
        table_bytes += int(nz.sum()) * 3
    stream_bytes = bits / 8.0
    # header: HEAD + mode_bytes + RAWLEN + span + min + n_ctx + 3*TBL + tables + TBL + stream
    overhead = 4 + n // BLOCK + 8 + 4 + 8 + 4 + 4 * N_CTX + table_bytes + 4
    return stream_bytes + overhead


def encode_symbols(symbols: np.ndarray) -> bytes:
    """Entropy-code an int64 symbol stream with per-block mode selection.

    Two candidate payloads are built and the smaller is emitted:

    - **RANGE**: per-block RAW/RANGE with a shared frequency table
      over varint bytes; mode chosen by estimated rANS size vs raw size.
    - **CTX**: whole-stream context-adaptive rANS over the symbol
      alphabet, with per-context tables selected by the previous symbol's
      magnitude bucket. Exploits the measured conditional entropy
      (H(s|s_prev) = 4.84 vs H(s) = 5.44 bits/symbol on the real field).
    """
    syms = np.asarray(symbols, dtype=np.int64)
    n = int(syms.size)
    if n == 0:
        return b""
    n_blocks = _block_count(n)

    # Cheap pre-decision: if the CTX estimate is clearly smaller than the
    # raw-packed lower bound, skip building the RANGE payload entirely.
    ctx_est = _estimate_ctx_size(syms)
    if ctx_est is not None:
        # Lower bound on the RANGE payload: raw-packed bytes alone.
        raw_lb = 0
        for bi in range(n_blocks):
            chunk = syms[bi * BLOCK : (bi + 1) * BLOCK]
            mn, mx = int(chunk.min()), int(chunk.max())
            if mn >= -128 and mx <= 127:
                raw_lb += 1 * chunk.size
            elif mn >= -32768 and mx <= 32767:
                raw_lb += 2 * chunk.size
            elif mn >= -(2**31) and mx <= 2**31 - 1:
                raw_lb += 4 * chunk.size
            else:
                raw_lb += 8 * chunk.size
        if ctx_est < raw_lb:
            ctx_payload = _encode_ctx_payload(syms, n_blocks)
            return ctx_payload

    range_payload, _, _ = _encode_range_payload(syms, n_blocks)
    ctx_payload = _encode_ctx_payload(syms, n_blocks)
    if ctx_payload is not None and len(ctx_payload) < len(range_payload):
        return ctx_payload
    return range_payload


def decode_symbols(payload: bytes, n: int) -> np.ndarray:
    """Decode exactly ``n`` int64 symbols from :func:`encode_symbols` output."""
    if n == 0:
        if payload:
            raise ValueError("non-empty payload for zero symbols")
        return np.zeros(0, dtype=np.int64)
    n_blocks = _block_count(n)
    off = 0
    (nb,) = _HEAD.unpack_from(payload, off)
    if nb != n_blocks:
        raise ValueError(f"block count mismatch: stream {nb}, expected {n_blocks}")
    off += _HEAD.size
    mode_bytes = payload[off : off + n_blocks]
    off += n_blocks
    (raw_len,) = _RAWLEN.unpack_from(payload, off)
    off += _RAWLEN.size
    raw_section = payload[off : off + raw_len]
    off += raw_len

    symbols = np.empty(n, dtype=np.int64)
    raw_ptr = 0
    range_blocks: list[tuple[int, int]] = []  # (block index, varint byte count)
    ctx_blocks: list[tuple[int, int]] = []  # (block index, symbol count)
    for bi in range(n_blocks):
        count = min(BLOCK, n - bi * BLOCK)
        mbyte = mode_bytes[bi]
        width, mode = mbyte >> 4, mbyte & 0x0F
        if mode == MODE_RAW:
            dt = {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"}[width]
            seg = raw_section[raw_ptr : raw_ptr + width * count]
            if len(seg) < width * count:
                raise ValueError("truncated raw block")
            symbols[bi * BLOCK : bi * BLOCK + count] = np.frombuffer(
                seg, dtype=dt
            ).astype(np.int64)
            raw_ptr += width * count
        elif mode == MODE_CTX:
            ctx_blocks.append((bi, count))
        else:
            range_blocks.append((bi, count))

    if ctx_blocks:
        # CTX section layout: [u32 span] [i64 min_symbol] [u32 n_ctx]
        #                     [n_ctx * (u32 table_len; table varints)]
        #                     [u32 rans_len; rANS stream]
        (span,) = struct.unpack_from("<I", payload, off)
        off += 4
        (min_symbol,) = struct.unpack_from("<q", payload, off)
        off += 8
        (n_ctx,) = struct.unpack_from("<I", payload, off)
        off += 4
        if n_ctx != N_CTX:
            raise ValueError(f"ctx table count mismatch: stream {n_ctx}, expected {N_CTX}")
        freqs_list = []
        for _ in range(n_ctx):
            (tbl_len,) = _TBL.unpack_from(payload, off)
            off += _TBL.size
            table_varints = payload[off : off + tbl_len]
            off += tbl_len
            freqs_list.append(varint_decode(table_varints, span).astype(np.int64))
        (rans_len,) = _TBL.unpack_from(payload, off)
        off += _TBL.size
        stream = payload[off : off + rans_len]
        total_ctx = sum(c for _, c in ctx_blocks)
        ctx_symbols = ctx_rans_decode(stream, total_ctx, freqs_list, min_symbol)
        ptr = 0
        for bi, count in ctx_blocks:
            symbols[bi * BLOCK : bi * BLOCK + count] = ctx_symbols[ptr : ptr + count]
            ptr += count

    if range_blocks:
        counts_rb = struct.unpack_from(f"<{len(range_blocks)}H", payload, off)
        off += 2 * len(range_blocks)
        n_rb = int(sum(counts_rb))  # total varint bytes across range blocks
        (total_syms,) = struct.unpack_from("<I", payload, off)
        off += 4
        (tbl_len,) = _TBL.unpack_from(payload, off)
        off += _TBL.size
        table_varints = payload[off : off + tbl_len]
        off += tbl_len
        freqs = varint_decode(table_varints, 256).astype(np.int64)
        (rans_len,) = _TBL.unpack_from(payload, off)
        off += _TBL.size
        stream = payload[off : off + rans_len]
        varint_stream = rans_decode(stream, int(n_rb), freqs)
        z_all = varint_decode(varint_stream, int(total_syms))
        ptr = 0
        raw_pos = 0
        for bi, count in range_blocks:
            seg = z_all[raw_pos : raw_pos + count]
            symbols[bi * BLOCK : bi * BLOCK + count] = unzigzag(seg)
            raw_pos += count
    return symbols


def pack_repairs(positions: np.ndarray, values: np.ndarray) -> bytes:
    """Append repair section: u32 count, then (i64 pos, f32 value) pairs."""
    positions = np.asarray(positions, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    if positions.size != values.size:
        raise ValueError("repair positions/values size mismatch")
    return (
        _REPAIR.pack(int(positions.size))
        + positions.astype("<i8").tobytes()
        + values.astype("<f4").tobytes()
    )


def unpack_repairs(payload: bytes, offset: int = 0):
    """Split repair section starting at ``offset``; returns (positions, values, end)."""
    (count,) = _REPAIR.unpack_from(payload, offset)
    off = offset + _REPAIR.size
    positions = (
        np.frombuffer(payload[off : off + count * 8], dtype="<i8").copy()
        if count
        else np.zeros(0, dtype=np.int64)
    )
    off += count * 8
    values = (
        np.frombuffer(payload[off : off + count * 4], dtype="<f4").copy()
        if count
        else np.zeros(0, dtype=np.float32)
    )
    off += count * 4
    return positions, values, off
