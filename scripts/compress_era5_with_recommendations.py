"""Compress an ERA5 pressure-level dataset using typed recommendations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from brace_compressor import BraceCodec, recommend_error_bound
from brace_compressor.container import read_container


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the ERA5 recommendation example."""

    parser = argparse.ArgumentParser(
        description="Compress ERA5 variables using compression-recommendations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/era5_pressure_20260715T1200_4levels.nc"),
        help="Input ERA5 NetCDF dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/era5_pressure_recommendation_example.json"),
        help="JSON file for benchmark results.",
    )
    parser.add_argument(
        "--variable",
        action="append",
        dest="variables",
        help="Variable to process; repeat to select multiple variables (default: all).",
    )
    return parser.parse_args()


def measure_variable(name: str, data_array: xr.DataArray) -> dict[str, object]:
    """Compress and validate one ERA5 variable using its recommendation."""

    original = np.ascontiguousarray(data_array.values)
    recommendation = recommend_error_bound(name, level_kind="pressure")
    codec = BraceCodec.from_recommendation(
        shape=original.shape,
        variable=name,
        dtype=original.dtype,
    )

    encode_start = time.perf_counter()
    encoded = codec.encode(original)
    encode_seconds = time.perf_counter() - encode_start
    decode_start = time.perf_counter()
    decoded = codec.decode(encoded)
    decode_seconds = time.perf_counter() - decode_start
    header = read_container(encoded).header_extra
    diagnostics = header.get("recommendation_checks", [])
    requirements_passed = all(
        bool(check.get("passed", False)) for check in diagnostics
    )

    finite = np.isfinite(original)
    difference = np.abs(decoded.astype(np.float64) - original.astype(np.float64))
    if recommendation.mode == "relative":
        denominator = np.abs(original.astype(np.float64))
        error = np.divide(
            difference,
            denominator,
            out=np.zeros_like(difference),
            where=finite & (denominator != 0),
        )
        violations = finite & (original != 0) & (error > recommendation.value)
        max_error = float(error[finite & (original != 0)].max(initial=0.0))
    else:
        error = difference
        violations = finite & (error > recommendation.value)
        max_error = float(error[finite].max(initial=0.0))

    return {
        "variable": name,
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "bound_mode": recommendation.mode,
        "bound": recommendation.value,
        "input_bytes": int(original.nbytes),
        "encoded_bytes": len(encoded),
        "compression_ratio": original.nbytes / len(encoded),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "max_error": max_error,
        "scalar_reference_violations": int(violations.sum()),
        "zero_mismatches": int(np.count_nonzero((original == 0) != (decoded == 0))),
        "selected_requirements": header.get("recommendation_plan", {}).get("selected", []),
        "strategy": header.get("strategy"),
        "n_repaired": int(header.get("n_repaired", 0)),
        "recommendation_checks": diagnostics,
        "all_requirements_passed": requirements_passed,
        "fallback": header.get("strategy") in {"absolute-scalar", "relative-scalar"},
    }


def main() -> None:
    """Run the recommendation-driven ERA5 compression example."""

    args = parse_args()
    selected = set(args.variables) if args.variables else None
    results: list[dict[str, object]] = []
    with xr.open_dataset(args.input, engine="h5netcdf") as dataset:
        for name, data_array in dataset.data_vars.items():
            if selected is not None and name not in selected:
                continue
            result = measure_variable(name, data_array)
            results.append(result)
            print(
                f"{name:4s} {result['bound_mode']:8s} "
                f"bound={result['bound']:g} "
                f"CR={result['compression_ratio']:.3f}x "
                f"max_error={result['max_error']:.6g} "
                f"scalar_violations={result['scalar_reference_violations']} "
                f"strategy={result['strategy']} "
                f"requirements={'pass' if result['all_requirements_passed'] else 'FAIL'}"
            )

    if not results:
        raise SystemExit("No variables selected or found in the dataset")
    failed = [result["variable"] for result in results if not result["all_requirements_passed"]]
    if failed:
        raise SystemExit(f"Recommendation requirements failed for: {', '.join(failed)}")
    total_input = sum(int(result["input_bytes"]) for result in results)
    total_encoded = sum(int(result["encoded_bytes"]) for result in results)
    output = {
        "input": str(args.input),
        "variables": results,
        "overall_compression_ratio": total_input / total_encoded,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Overall CR={output['overall_compression_ratio']:.3f}x")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
