#!/usr/bin/env python3
"""Compress/decompress a HOAPS-wvpa-like field with BRACE and display statistics.

Utility CLI for the `brace-compressor` codec. Run from the repository
root with the project venv:

    .venv/bin/python scripts/compress_stats.py
    .venv/bin/python scripts/compress_stats.py --shape 8 90 180 --bound 0.05
    .venv/bin/python scripts/compress_stats.py --input field.nc --bound 0.01
    .venv/bin/python scripts/compress_stats.py --sweep 0.01 0.05 0.2 --shape 4 32 64

The script needs no real HOAPS data: without ``--input`` it generates a
smooth space-time synthetic field (deterministic seed) with a land-strip
missing mask, mimicking HOAPS wvpa statistics. With ``--input`` it loads a
``.npy`` or NetCDF file. NetCDF input uses the ``wvpa`` variable by default;
2-D lat×lon input is promoted to a single time slice.

Reported statistics include sizes and compression ratio (CR), the
verified maximum absolute error vs the configured bound (SC-001),
missing-mask fidelity (SC-002), per-value statistics (RMSE, bias),
timing, and the in-container metrics recorded by the codec (FR-012).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make `src/` importable when running straight from a checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from brace_compressor import BraceCodec  # noqa: E402
from brace_compressor.container import read_container  # noqa: E402


# ---------------------------------------------------------------------------
# Field generation / loading
# ---------------------------------------------------------------------------
def generate_synthetic_field(
    shape: tuple[int, int, int], seed: int, missing_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth space-time field with a land-strip + scattered missing mask."""
    t, lat, lon = shape
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, lon)
    y = np.linspace(0.0, np.pi, lat)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    field = np.empty(shape, dtype=np.float32)
    for ti in range(t):
        phase = 0.35 * ti
        base = 25.0 + 12.0 * np.sin(yy + phase) * np.cos(xx - 0.5 * phase)
        detail = rng.normal(0.0, 0.8, size=(lat, lon))
        field[ti] = (base + detail).astype(np.float32)

    mask = np.zeros(shape, dtype=bool)
    if missing_frac > 0:
        strip = max(1, int(lat * missing_frac * 0.8))
        mask[:, :strip, :] = True
        scatter = rng.random(shape) > (1.0 - missing_frac * 0.35)
        mask |= scatter
    return np.where(mask, np.nan, field).astype(np.float32), mask


def load_field(path: Path, variable: str = "wvpa") -> np.ndarray:
    """Load a NumPy or NetCDF field; promote 2-D input to one time slice."""
    if path.suffix.lower() in {".nc", ".nc4", ".netcdf"}:
        try:
            import xarray as xr
        except ImportError as exc:
            raise SystemExit(
                "NetCDF input requires xarray and h5netcdf; install with "
                "`.venv/bin/python -m pip install -e '.[analysis]'`"
            ) from exc
        try:
            with xr.open_dataset(path, engine="h5netcdf") as dataset:
                if variable not in dataset:
                    available = ", ".join(dataset.data_vars)
                    raise SystemExit(
                        f"NetCDF variable {variable!r} not found; available: {available}"
                    )
                field = dataset[variable].values.astype(np.float32)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"could not read NetCDF input {path}: {exc}") from exc
    else:
        field = np.load(path).astype(np.float32)
    if field.ndim == 2:
        field = field[None, ...]
    if field.ndim != 3:
        raise SystemExit(f"input must be 2-D or 3-D; got {field.shape}")
    return np.ascontiguousarray(field)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024 or unit == "GB":
            return f"{nbytes:,.0f} {unit}" if unit == "B" else f"{nbytes:,.2f} {unit}"
        nbytes /= 1024.0


def field_stats(field: np.ndarray) -> dict:
    valid = ~np.isnan(field)
    vals = field[valid].astype(np.float64)
    return {
        "n_cells": int(field.size),
        "n_valid": int(valid.sum()),
        "n_missing": int(np.isnan(field).sum()),
        "missing_pct": 100.0 * np.isnan(field).sum() / field.size,
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "std": float(vals.std()) if vals.size else float("nan"),
    }


def run_roundtrip(
    field: np.ndarray, bound: float, outer_compress: bool = True, quiet: bool = False
) -> dict:
    """Encode + decode one field at ``bound``; collect all statistics."""
    shape = field.shape
    missing_value = "nan" if np.isnan(field).any() else 0.0
    if missing_value == 0.0:
        raise SystemExit(
            "input contains no missings; a sentinel would corrupt valid zeros. "
            "Use NaN-missing input (HOAPS wvpa uses a fill/NaN sentinel)."
        )

    codec = BraceCodec(
        shape=shape,
        error_bound=bound,
        missing_value=missing_value,
        outer_compress=outer_compress,
    )

    t0 = time.perf_counter()
    encoded = codec.encode(field)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = codec.decode(encoded)
    t_dec = time.perf_counter() - t0

    # --- Verification statistics (independent of the codec's own claims) --
    orig_valid = ~np.isnan(field)
    dec_valid = ~np.isnan(decoded)
    mask_ok = bool(np.array_equal(orig_valid, dec_valid))
    mask_sentinel_ok = None
    if missing_value != "nan":
        mask_sentinel_ok = bool(
            np.array_equal(field == np.float32(missing_value), decoded == np.float32(missing_value))
        )

    err = np.abs(decoded[orig_valid].astype(np.float64) - field[orig_valid].astype(np.float64))
    signed_err = decoded[orig_valid].astype(np.float64) - field[orig_valid].astype(np.float64)
    max_abs_error = float(err.max()) if err.size else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
    bias = float(signed_err.mean()) if err.size else 0.0

    # Container header metrics (recorded by the codec, FR-012)
    header = read_container(encoded).header_extra
    meta = header.get("metrics", {})

    orig_size = int(field.nbytes)
    comp_size = len(encoded)
    stats = {
        "bound": bound,
        "shape": shape,
        "dtype": str(field.dtype),
        "field": field_stats(field),
        "orig_size": orig_size,
        "comp_size": comp_size,
        "cr": orig_size / comp_size if comp_size else float("inf"),
        "reduction_pct": 100.0 * (1.0 - comp_size / orig_size) if orig_size else 0.0,
        "bits_per_value": 8.0 * comp_size / max(1, field.size),
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "bias": bias,
        "p99_abs_error": float(np.percentile(err, 99)) if err.size else 0.0,
        "violations": int((err > bound).sum()) if bound > 0 else 0,
        "bound_respected": bool(max_abs_error <= bound) or bound == 0,
        "mask_identical": mask_ok,
        "mask_sentinel_ok": mask_sentinel_ok,
        "encode_s": t_enc,
        "decode_s": t_dec,
        "throughput_mbs": (orig_size / 1e6) / t_enc if t_enc > 0 else float("nan"),
        "header_mode": header.get("mode"),
        "header_quant_step": header.get("quant_step"),
        "container_metrics": meta,
        "outer_compressed": read_container(encoded).outer_compressed,
    }
    return stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_report(stats: dict) -> None:
    f = stats["field"]
    line = "─" * 66
    print()
    print(line)
    print(
        f" HOAPS wvpa compression report  ·  bound={stats['bound']:g} · "
        f"shape={stats['shape'][0]}×{stats['shape'][1]}×{stats['shape'][2]} · "
        f"{stats['dtype']}"
    )
    print(line)

    print(" Input field")
    print(
        f"   cells {f['n_cells']:,}   valid {f['n_valid']:,}   "
        f"missing {f['n_missing']:,} ({f['missing_pct']:.1f} %)"
    )
    print(
        f"   range [{f['min']:.3f}, {f['max']:.3f}]  mean {f['mean']:.3f}  "
        f"std {f['std']:.3f}"
    )

    print(" Size")
    print(f"   original          {human(stats['orig_size']):>12}")
    print(f"   compressed        {human(stats['comp_size']):>12}")
    print(
        f"   compression ratio {stats['cr']:>12.2f}×   "
        f"({stats['reduction_pct']:.1f} % smaller, "
        f"{stats['bits_per_value']:.2f} bits/value)"
    )

    print(" Accuracy")
    if stats["bound"] == 0:
        print("   bound 0: exact raw-float32 encoding, no quantization")
    else:
        status = "OK " if stats["bound_respected"] else "FAIL"
        print(
            f"   [{status}] max |error| {stats['max_abs_error']:.6f} ≤ bound "
            f"{stats['bound']:g}   ({stats['violations']} violations)"
        )
    print(f"   RMSE  {stats['rmse']:.6f}")
    print(f"   bias  {stats['bias']:+.6f}")
    print(f"   p99   |error| {stats['p99_abs_error']:.6f}")
    mask_note = "identical" if stats["mask_identical"] else "MISMATCH"
    print(f"   missing mask {mask_note} (SC-002)")

    print(" Timing")
    print(f"   encode {stats['encode_s']:.2f} s   decode {stats['decode_s']:.2f} s")
    print(f"   throughput {stats['throughput_mbs']:.3f} MB/s (encode)")

    m = stats["container_metrics"]
    if m:
        print(" Codec-reported metrics (container header, FR-012)")
        print(
            f"   max_abs_error {m.get('max_abs_error', float('nan')):.6f} · "
            f"n_repaired {m.get('n_repaired', '?')} · "
            f"payload {human(m.get('payload_size', 0))} · "
            f"bound_respected {m.get('bound_respected')}"
        )
        print(
            f"   quant_step {stats['header_quant_step']} · "
            f"mode {'quantized' if stats['header_mode'] == 0 else 'exact'} · "
            f"outer_compressed {stats['outer_compressed']}"
        )
    print(line)


def print_sweep(sweep_rows: list[dict]) -> None:
    print()
    print("─" * 78)
    print(" Bound sweep — CR vs accuracy trade-off (SC-004)")
    print("─" * 78)
    hdr = (
        f"{'bound':>7} | {'orig':>9} | {'compressed':>10} | {'CR':>6} | "
        f"{'red.%':>6} | {'max|err|':>8} | {'OK':>3} | {'enc':>5} | {'dec':>5}"
    )
    print(hdr)
    print("─" * 78)
    for s in sweep_rows:
        print(
            f"{s['bound']:>7g} | {human(s['orig_size']):>9} | "
            f"{human(s['comp_size']):>10} | {s['cr']:>5.2f}× | "
            f"{s['reduction_pct']:>5.1f} | {s['max_abs_error']:>8.5f} | "
            f"{'OK' if s['bound_respected'] else 'FAIL':>3} | "
            f"{s['encode_s']:>5.2f} | {s['decode_s']:>5.2f}"
        )
    crs = [r["cr"] for r in sweep_rows]
    # Bounds are listed in the order provided; each looser bound must not
    # compress worse than the previous (SC-004: CR non-decreasing).
    mono = all(a <= b * 1.02 for a, b in zip(crs, crs[1:])) if len(crs) > 1 else True
    print("─" * 78)
    print(f" CR non-decreasing for looser bounds (SC-004): {mono}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Compress + decompress a HOAPS-wvpa-like field with BRACE and report CR/statistics."
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Optional .npy or NetCDF field; default: synthetic HOAPS-like field",
    )
    p.add_argument(
        "--variable", default="wvpa",
        help="NetCDF variable to compress (default: wvpa)",
    )
    p.add_argument(
        "--shape", type=int, nargs=3, metavar=("T", "LAT", "LON"), default=[8, 90, 180],
        help="Synthetic field shape (default: 8 90 180)",
    )
    p.add_argument(
        "--bound", type=float, default=0.05,
        help="Absolute error bound (default 0.05); 0 = exact raw path",
    )
    p.add_argument(
        "--sweep", type=float, nargs="*", default=None,
        help="Run several bounds and print a comparison table (overrides --bound display style)",
    )
    p.add_argument("--seed", type=int, default=42, help="Synthetic field seed (default 42)")
    p.add_argument(
        "--missing-frac", type=float, default=0.12,
        help="Fraction of missing cells for the synthetic mask (default 0.12)",
    )
    p.add_argument(
        "--no-outer-compress", action="store_true",
        help="Disable the lossless outer Zstandard pass on payloads",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print the stats as JSON instead of the formatted report",
    )
    args = p.parse_args()

    if args.input is not None:
        field = load_field(args.input, variable=args.variable)
    else:
        field, _ = generate_synthetic_field(
            tuple(args.shape), seed=args.seed, missing_frac=args.missing_frac
        )

    bounds = args.sweep if args.sweep else [args.bound]
    if 0 in bounds and len(bounds) > 1:
        print("note: bound 0 uses exact raw-float32 encoding", file=sys.stderr)

    rows = [
        run_roundtrip(field, b, outer_compress=not args.no_outer_compress, quiet=args.json)
        for b in bounds
    ]

    if args.json:
        print(
            json.dumps(
                {"n_runs": len(rows), "results": rows},
                indent=2,
                default=lambda o: o if isinstance(o, (int, float, str, bool, list, dict, type(None))) else str(o),
            )
        )
        return

    if len(rows) == 1:
        print_report(rows[0])
    else:
        print_sweep(rows)
        print()
        print_report(max(rows, key=lambda r: r["cr"]))


if __name__ == "__main__":
    main()
