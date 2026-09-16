# Implementation Plan: HOAPS WVPA Causal Compressor

**Branch**: `remove-transformer`

## Design

The package exposes `HoapsWvpaCodec` under `src/hoaps_compressor/`. Encoding extracts and separately compresses the missing mask, walks valid cells in deterministic order, quantizes residuals using the configured bound, verifies the decoder simulation, and writes a versioned container. Decoding repeats the same walk and restores the sentinel from the mask.

The reconstructed-neighbor stencil is longitude-local: left 8, top 2, top-left 1, top-right 1, and temporal parent 1. A constant `32.0` cold-start value is deterministic at both ends. `model/entropy.py` contains only lossless symbol coding. The optional Rust extension mirrors the Python scan.

## Validation

Run unit tests for bounds, quantization, masks, containers, and entropy coding; contract tests for `numcodecs`; and integration tests for error bounds, missing values, configurable bounds, and space-time compression. Run the full suite with `pytest tests -q`.
