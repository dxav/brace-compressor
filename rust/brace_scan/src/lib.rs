//! Rust-accelerated scan for the BRACE codec.
//!
//! This is a bit-exact port of the pure-Python scan in
//! `src/brace_compressor/codec.py` (`_causal_scan_encode` /
//! `_causal_scan_decode`). The scan order, neighbor weights and
//! arithmetic are identical so encode/decode remain bit-consistent.
//!
//! The scan is the dominant cost of compression/decompression; moving it
//! to Rust removes the per-cell Python interpreter overhead.

use numpy::{
    ndarray::ArrayView3, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray3,
};
use pyo3::prelude::*;

fn validate_step(step: f64, function: &str) -> PyResult<()> {
    if step <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be > 0 in {function}"
        )));
    }
    Ok(())
}

fn predict(
    recon_rows: &[f64],
    lat: usize,
    lon: usize,
    row: usize,
    y_index: usize,
    time_index: usize,
    x_index: usize,
    prior: f64,
) -> f64 {
    let has_top = y_index > 0;
    let top_row = row.saturating_sub(1);
    let has_time = time_index > 0;
    let time_row = row.saturating_sub(lat);
    let mut predictions = [0.0; 5];
    let mut weights = [0.0; 5];
    let mut neighbor_count = 0;
    let mut add_neighbor = |value: f64, weight: f64| {
        predictions[neighbor_count] = value;
        weights[neighbor_count] = weight;
        neighbor_count += 1;
    };
    if x_index > 0 && recon_rows[row * lon + x_index - 1] != 0.0 {
        add_neighbor(recon_rows[row * lon + x_index - 1], 8.0);
    }
    if has_top && recon_rows[top_row * lon + x_index] != 0.0 {
        add_neighbor(recon_rows[top_row * lon + x_index], 2.0);
    }
    if has_top && x_index > 0 && recon_rows[top_row * lon + x_index - 1] != 0.0 {
        add_neighbor(recon_rows[top_row * lon + x_index - 1], 1.0);
    }
    if has_top && x_index + 1 < lon && recon_rows[top_row * lon + x_index + 1] != 0.0 {
        add_neighbor(recon_rows[top_row * lon + x_index + 1], 1.0);
    }
    if has_time && recon_rows[time_row * lon + x_index] != 0.0 {
        add_neighbor(recon_rows[time_row * lon + x_index], 1.0);
    }
    if neighbor_count == 0 {
        prior
    } else {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for neighbor_index in 0..neighbor_count {
            weighted_sum += predictions[neighbor_index] * weights[neighbor_index];
            weight_sum += weights[neighbor_index];
        }
        weighted_sum / weight_sum
    }
}

fn encode_scan<'py, T>(
    py: Python<'py>,
    field: ArrayView3<'_, T>,
    mask: ArrayView3<'_, bool>,
    prior: ArrayView3<'_, T>,
    step: f64,
    function: &str,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)>
where
    T: Copy + Into<f64>,
{
    let shape = field.shape();
    if mask.shape() != shape || prior.shape() != shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "field/mask/prior shape mismatch",
        ));
    }
    validate_step(step, function)?;
    let (t, lat, lon) = (shape[0], shape[1], shape[2]);
    let n_valid = mask.iter().filter(|value| !**value).count();
    let mut symbols = vec![0i64; n_valid];
    let mut recon_rows = vec![0.0f64; t * lat * lon];
    let inv_step = 1.0 / step;
    let lo = -(1i64 << 31) + 1;
    let hi = (1i64 << 31) - 1;
    let mut symbol_index = 0;

    for time_index in 0..t {
        let base = time_index * lat;
        for y_index in 0..lat {
            let row = base + y_index;
            for x_index in 0..lon {
                if mask[[time_index, y_index, x_index]] {
                    continue;
                }
                let prediction = predict(
                    &recon_rows,
                    lat,
                    lon,
                    row,
                    y_index,
                    time_index,
                    x_index,
                    prior[[time_index, y_index, x_index]].into(),
                );
                let residual = field[[time_index, y_index, x_index]].into() - prediction;
                let symbol = ((residual * inv_step + 0.5).floor() as i64).clamp(lo, hi);
                symbols[symbol_index] = symbol;
                recon_rows[row * lon + x_index] = prediction + symbol as f64 * step;
                symbol_index += 1;
            }
        }
    }
    let symbols = symbols.into_pyarray_bound(py);
    let reconstruction = recon_rows.into_pyarray_bound(py).reshape([t * lat, lon])?;
    Ok((symbols, reconstruction))
}

fn decode_scan<'py, T>(
    py: Python<'py>,
    prior: ArrayView3<'_, T>,
    mask: ArrayView3<'_, bool>,
    symbols: numpy::ndarray::ArrayView1<'_, i64>,
    step: f64,
    origin: f64,
    function: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    T: Copy + Into<f64>,
{
    let shape = prior.shape();
    if mask.shape() != shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prior/mask shape mismatch",
        ));
    }
    validate_step(step, function)?;
    let (t, lat, lon) = (shape[0], shape[1], shape[2]);
    let mut recon_rows = vec![0.0f64; t * lat * lon];
    let mut symbol_index = 0;

    for time_index in 0..t {
        let base = time_index * lat;
        for y_index in 0..lat {
            let row = base + y_index;
            for x_index in 0..lon {
                if mask[[time_index, y_index, x_index]] {
                    continue;
                }
                let prediction = predict(
                    &recon_rows,
                    lat,
                    lon,
                    row,
                    y_index,
                    time_index,
                    x_index,
                    prior[[time_index, y_index, x_index]].into(),
                );
                recon_rows[row * lon + x_index] =
                    prediction + symbols[symbol_index] as f64 * step + origin;
                symbol_index += 1;
            }
        }
    }
    Ok(recon_rows.into_pyarray_bound(py).reshape([t * lat, lon])?)
}

/// Encode-side scan.
///
/// Args:
///   field:  float32 [T, lat, lon] original values
///   mask:   bool   [T, lat, lon] (True = missing)
///   prior:  float32 [T, lat, lon] deterministic cold-start prior
///   step:   float quantization step (> 0)
///
/// Returns (symbols int64 [n_valid], recon_rows float64 [T*lat, lon]).
#[pyfunction]
#[pyo3(signature = (field, mask, prior, step))]
fn causal_scan_encode<'py>(
    py: Python<'py>,
    field: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray3<'py, bool>,
    prior: PyReadonlyArray3<'py, f32>,
    step: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    encode_scan(
        py,
        field.as_array(),
        mask.as_array(),
        prior.as_array(),
        step,
        "causal_scan_encode",
    )
}

/// Float64 variant of the encode-side scan. Kept separate from the legacy
/// float32 entry point so existing callers retain the same ABI and arithmetic.
#[pyfunction]
#[pyo3(signature = (field, mask, prior, step))]
fn causal_scan_encode_f64<'py>(
    py: Python<'py>,
    field: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    prior: PyReadonlyArray3<'py, f64>,
    step: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    encode_scan(
        py,
        field.as_array(),
        mask.as_array(),
        prior.as_array(),
        step,
        "causal_scan_encode_f64",
    )
}

/// Decode-side scan (mirror of encode; same order/math).
///
/// Args:
///   prior:   float32 [T, lat, lon]
///   mask:    bool   [T, lat, lon]
///   symbols: int64  [n_valid]
///   step:    float
///   origin:  float (always 0.0 in current codec)
///
/// Returns recon_rows float64 [T*lat, lon].
#[pyfunction]
#[pyo3(signature = (prior, mask, symbols, step, origin))]
fn causal_scan_decode<'py>(
    py: Python<'py>,
    prior: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray3<'py, bool>,
    symbols: PyReadonlyArray1<'py, i64>,
    step: f64,
    origin: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    decode_scan(
        py,
        prior.as_array(),
        mask.as_array(),
        symbols.as_array(),
        step,
        origin,
        "causal_scan_decode",
    )
}

/// Float64 variant of the decode-side scan, mirroring `causal_scan_encode_f64`.
#[pyfunction]
#[pyo3(signature = (prior, mask, symbols, step, origin))]
fn causal_scan_decode_f64<'py>(
    py: Python<'py>,
    prior: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray3<'py, bool>,
    symbols: PyReadonlyArray1<'py, i64>,
    step: f64,
    origin: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    decode_scan(
        py,
        prior.as_array(),
        mask.as_array(),
        symbols.as_array(),
        step,
        origin,
        "causal_scan_decode_f64",
    )
}

/// Python module definition.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(causal_scan_encode, m)?)?;
    m.add_function(wrap_pyfunction!(causal_scan_encode_f64, m)?)?;
    m.add_function(wrap_pyfunction!(causal_scan_decode, m)?)?;
    m.add_function(wrap_pyfunction!(causal_scan_decode_f64, m)?)?;
    Ok(())
}
