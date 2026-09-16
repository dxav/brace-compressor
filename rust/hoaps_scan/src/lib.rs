//! Rust-accelerated causal scan for the HOAPS wvpa codec.
//!
//! This is a bit-exact port of the pure-Python causal scan in
//! `src/hoaps_compressor/codec.py` (`_causal_scan_encode` /
//! `_causal_scan_decode`). The scan order, neighbor weights and
//! arithmetic are identical so encode/decode remain bit-consistent.
//!
//! The scan is the dominant cost of compression/decompression; moving it
//! to Rust removes the per-cell Python interpreter overhead.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Encode-side causal scan.
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
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let f = field.as_array();
    let m = mask.as_array();
    let p = prior.as_array();
    let shp = f.shape();
    let t = shp[0];
    let lat = shp[1];
    let lon = shp[2];
    let mshp = m.shape();
    let pshp = p.shape();
    if mshp != shp || pshp != shp {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "field/mask/prior shape mismatch",
        ));
    }
    if step <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step must be > 0 in causal_scan_encode",
        ));
    }

    // Count valid cells.
    let mut n_valid: usize = 0;
    for ti in 0..t {
        for yi in 0..lat {
            for xi in 0..lon {
                if !m[[ti, yi, xi]] {
                    n_valid += 1;
                }
            }
        }
    }

    let mut symbols = vec![0i64; n_valid];
    // recon_rows indexed [row = ti*lat + yi][xi]
    let mut recon_rows = vec![0f64; t * lat * lon];

    let inv_step = 1.0 / step;
    let lo = -(1i64 << 31) + 1;
    let hi = (1i64 << 31) - 1;

    let mut k: usize = 0;
    for ti in 0..t {
        let base = ti * lat;
        for yi in 0..lat {
            let row = base + yi;
            let has_top = yi > 0;
            let top_row = row - 1;
            let has_time = ti > 0;
            let time_row = row - lat;
            for xi in 0..lon {
                if m[[ti, yi, xi]] {
                    continue;
                }
                // causal neighbors (same order/weights as Python).
                // Spatial-weighted stencil (T047): left 5, top 5,
                // top-left 2, top-right 2, temporal parent 1.
                let mut preds: [f64; 5] = [0.0; 5];
                let mut wts: [f64; 5] = [0.0; 5];
                let mut n = 0usize;
                if xi > 0 && recon_rows[row * lon + (xi - 1)] != 0.0 {
                    preds[n] = recon_rows[row * lon + (xi - 1)];
                    wts[n] = 5.0;
                    n += 1;
                }
                if has_top && recon_rows[top_row * lon + xi] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + xi];
                    wts[n] = 5.0;
                    n += 1;
                }
                if has_top && xi > 0 && recon_rows[top_row * lon + (xi - 1)] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + (xi - 1)];
                    wts[n] = 2.0;
                    n += 1;
                }
                if has_top && xi < lon - 1 && recon_rows[top_row * lon + (xi + 1)] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + (xi + 1)];
                    wts[n] = 2.0;
                    n += 1;
                }
                if has_time && recon_rows[time_row * lon + xi] != 0.0 {
                    preds[n] = recon_rows[time_row * lon + xi];
                    wts[n] = 1.0;
                    n += 1;
                }
                let pred: f64 = if n > 0 {
                    let mut acc = 0.0;
                    let mut wsum = 0.0;
                    for i in 0..n {
                        acc += preds[i] * wts[i];
                        wsum += wts[i];
                    }
                    acc / wsum
                } else {
                    p[[ti, yi, xi]] as f64
                };
                let r = f[[ti, yi, xi]] as f64 - pred;
                let mut q = (r * inv_step + 0.5).floor() as i64;
                if q < lo {
                    q = lo;
                } else if q > hi {
                    q = hi;
                }
                symbols[k] = q;
                recon_rows[row * lon + xi] = pred + q as f64 * step;
                k += 1;
            }
        }
    }

    let sym_arr = symbols.into_pyarray_bound(py);
    let recon_arr = recon_rows.into_pyarray_bound(py).reshape([t * lat, lon])?;
    Ok((sym_arr, recon_arr))
}

/// Decode-side causal scan (mirror of encode; same order/math).
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
    let p = prior.as_array();
    let m = mask.as_array();
    let sym = symbols.as_array();
    let shp = p.shape();
    let t = shp[0];
    let lat = shp[1];
    let lon = shp[2];
    let mshp = m.shape();
    if mshp != shp {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prior/mask shape mismatch",
        ));
    }
    if step <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "step must be > 0 in causal_scan_decode",
        ));
    }

    let mut recon_rows = vec![0f64; t * lat * lon];
    let mut k: usize = 0;
    for ti in 0..t {
        let base = ti * lat;
        for yi in 0..lat {
            let row = base + yi;
            let has_top = yi > 0;
            let top_row = row - 1;
            let has_time = ti > 0;
            let time_row = row - lat;
            for xi in 0..lon {
                if m[[ti, yi, xi]] {
                    continue;
                }
                let mut preds: [f64; 5] = [0.0; 5];
                let mut wts: [f64; 5] = [0.0; 5];
                let mut n = 0usize;
                if xi > 0 && recon_rows[row * lon + (xi - 1)] != 0.0 {
                    preds[n] = recon_rows[row * lon + (xi - 1)];
                    wts[n] = 5.0;
                    n += 1;
                }
                if has_top && recon_rows[top_row * lon + xi] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + xi];
                    wts[n] = 5.0;
                    n += 1;
                }
                if has_top && xi > 0 && recon_rows[top_row * lon + (xi - 1)] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + (xi - 1)];
                    wts[n] = 2.0;
                    n += 1;
                }
                if has_top && xi < lon - 1 && recon_rows[top_row * lon + (xi + 1)] != 0.0 {
                    preds[n] = recon_rows[top_row * lon + (xi + 1)];
                    wts[n] = 2.0;
                    n += 1;
                }
                if has_time && recon_rows[time_row * lon + xi] != 0.0 {
                    preds[n] = recon_rows[time_row * lon + xi];
                    wts[n] = 1.0;
                    n += 1;
                }
                let pred: f64 = if n > 0 {
                    let mut acc = 0.0;
                    let mut wsum = 0.0;
                    for i in 0..n {
                        acc += preds[i] * wts[i];
                        wsum += wts[i];
                    }
                    acc / wsum
                } else {
                    p[[ti, yi, xi]] as f64
                };
                let dq = sym[k] as f64 * step + origin;
                recon_rows[row * lon + xi] = pred + dq;
                k += 1;
            }
        }
    }

    Ok(recon_rows.into_pyarray_bound(py).reshape([t * lat, lon])?)
}

/// Python module definition.
#[pymodule]
fn hoaps_scan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(causal_scan_encode, m)?)?;
    m.add_function(wrap_pyfunction!(causal_scan_decode, m)?)?;
    Ok(())
}
