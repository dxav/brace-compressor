//! rANS entropy codecs exposed by the optional Rust extension.

use numpy::{
    ndarray::{ArrayView1, ArrayView2},
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const RANS_SCALE: u32 = 16;
const RANS_M: usize = 1 << RANS_SCALE;
const RANS_L: u64 = 1 << 23;
const CTX_BUCKET_1: i64 = 1;
const CTX_BUCKET_2: i64 = 8;

fn context_of(value: i64) -> usize {
    let magnitude = value.unsigned_abs();
    if magnitude <= CTX_BUCKET_1 as u64 {
        0
    } else if magnitude <= CTX_BUCKET_2 as u64 {
        1
    } else {
        2
    }
}

fn validate_freqs(freqs: ArrayView2<'_, i64>) -> PyResult<usize> {
    if freqs.nrows() != 3 || freqs.ncols() == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freqs must have shape (3, span) with span > 0",
        ));
    }
    for context in 0..3 {
        let mut total = 0i64;
        for symbol in 0..freqs.ncols() {
            let frequency = freqs[[context, symbol]];
            if frequency < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "frequency tables cannot contain negative values",
                ));
            }
            total += frequency;
        }
        if total != 0 && total != RANS_M as i64 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "each frequency table must sum to 0 or 65536",
            ));
        }
    }
    Ok(freqs.ncols())
}

fn cumulative_tables(freqs: ArrayView2<'_, i64>) -> Vec<Vec<u64>> {
    (0..3)
        .map(|context| {
            let mut cumulative = vec![0u64; freqs.ncols() + 1];
            for symbol in 0..freqs.ncols() {
                cumulative[symbol + 1] = cumulative[symbol] + freqs[[context, symbol]] as u64;
            }
            cumulative
        })
        .collect()
}

fn validate_static_freqs(freqs: &ArrayView1<'_, i64>) -> PyResult<Vec<u64>> {
    if freqs.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frequency table must contain 256 entries",
        ));
    }
    let mut cumulative = vec![0u64; 257];
    for symbol in 0..256 {
        let frequency = freqs[symbol];
        if frequency < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency table cannot contain negative values",
            ));
        }
        cumulative[symbol + 1] = cumulative[symbol] + frequency as u64;
    }
    if cumulative[256] != RANS_M as u64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frequency table must sum to 65536",
        ));
    }
    Ok(cumulative)
}

#[pyfunction]
#[pyo3(signature = (data, freqs))]
fn rans_encode<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyBytes>,
    freqs: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let data = data.as_bytes();
    if data.is_empty() {
        return Ok(PyBytes::new_bound(py, b""));
    }
    let freqs = freqs.as_array();
    let cumulative = validate_static_freqs(&freqs)?;
    let threshold = (RANS_L >> RANS_SCALE) << 8;
    let mut state = RANS_L;
    let mut emitted = Vec::with_capacity(data.len() * 2 + 4);
    for &byte in data.iter().rev() {
        let symbol = byte as usize;
        let frequency = freqs[symbol] as u64;
        if frequency == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency table has zero probability for a byte",
            ));
        }
        let x_max = threshold * frequency;
        while state >= x_max {
            emitted.push((state & 0xff) as u8);
            state >>= 8;
        }
        state = ((state / frequency) << RANS_SCALE) + (state % frequency) + cumulative[symbol];
    }
    for _ in 0..4 {
        emitted.push((state & 0xff) as u8);
        state >>= 8;
    }
    emitted.reverse();
    Ok(PyBytes::new_bound(py, &emitted))
}

#[pyfunction]
#[pyo3(signature = (src, n_bytes, freqs))]
fn rans_decode<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyBytes>,
    n_bytes: usize,
    freqs: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyBytes>> {
    if n_bytes == 0 {
        return Ok(PyBytes::new_bound(py, b""));
    }
    let data = src.as_bytes();
    if data.len() < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rANS stream is shorter than its initial state",
        ));
    }
    let freqs = freqs.as_array();
    let cumulative = validate_static_freqs(&freqs)?;
    let mut lookup = vec![0u8; RANS_M];
    for symbol in 0..256 {
        for slot in cumulative[symbol] as usize..cumulative[symbol + 1] as usize {
            lookup[slot] = symbol as u8;
        }
    }
    let mut state = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64;
    let mut data_index = 4;
    let mut output = vec![0u8; n_bytes];
    for byte in &mut output {
        let slot = (state & (RANS_M as u64 - 1)) as usize;
        let symbol = lookup[slot] as usize;
        let frequency = freqs[symbol] as u64;
        state = frequency * (state >> RANS_SCALE) + slot as u64 - cumulative[symbol];
        while state < RANS_L {
            if data_index >= data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "truncated rANS stream",
                ));
            }
            state = (state << 8) | data[data_index] as u64;
            data_index += 1;
        }
        *byte = symbol as u8;
    }
    Ok(PyBytes::new_bound(py, &output))
}

#[pyfunction]
#[pyo3(signature = (symbols, freqs, min_symbol))]
fn ctx_rans_encode<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, i64>,
    freqs: PyReadonlyArray2<'py, i64>,
    min_symbol: i64,
) -> PyResult<Bound<'py, PyBytes>> {
    let symbols = symbols.as_array();
    let freqs = freqs.as_array();
    let span = validate_freqs(freqs)?;
    let cumulative = cumulative_tables(freqs);
    let threshold = (RANS_L >> RANS_SCALE) << 8;
    let mut state = RANS_L;
    let mut emitted = Vec::with_capacity(symbols.len() * 2 + 4);

    for index in (0..symbols.len()).rev() {
        let symbol = symbols[index];
        let offset = symbol.checked_sub(min_symbol).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("symbol is outside the frequency span")
        })?;
        if offset < 0 || offset as usize >= span {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "symbol is outside the frequency span",
            ));
        }
        let symbol_index = offset as usize;
        let context = if index == 0 {
            0
        } else {
            context_of(symbols[index - 1])
        };
        let frequency = freqs[[context, symbol_index]] as u64;
        if frequency == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency table has zero probability for a symbol",
            ));
        }
        let x_max = threshold * frequency;
        while state >= x_max {
            emitted.push((state & 0xff) as u8);
            state >>= 8;
        }
        state = ((state / frequency) << RANS_SCALE)
            + (state % frequency)
            + cumulative[context][symbol_index];
    }
    for _ in 0..4 {
        emitted.push((state & 0xff) as u8);
        state >>= 8;
    }
    emitted.reverse();
    Ok(PyBytes::new_bound(py, &emitted))
}

#[pyfunction]
#[pyo3(signature = (src, n, freqs, min_symbol))]
fn ctx_rans_decode<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyBytes>,
    n: usize,
    freqs: PyReadonlyArray2<'py, i64>,
    min_symbol: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let data = src.as_bytes();
    if n == 0 {
        return Ok(vec![].into_pyarray_bound(py));
    }
    if data.len() < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rANS stream is shorter than its initial state",
        ));
    }
    let freqs = freqs.as_array();
    let span = validate_freqs(freqs)?;
    let cumulative = cumulative_tables(freqs);
    let mut lookup = vec![vec![0usize; RANS_M]; 3];
    for context in 0..3 {
        for symbol in 0..span {
            for slot in
                cumulative[context][symbol] as usize..cumulative[context][symbol + 1] as usize
            {
                lookup[context][slot] = symbol;
            }
        }
    }
    let mut state = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64;
    let mut data_index = 4;
    let mut output = vec![0i64; n];
    let mut previous = 0i64;
    for value in &mut output {
        let context = context_of(previous);
        let slot = (state & (RANS_M as u64 - 1)) as usize;
        let symbol_index = lookup[context][slot];
        let frequency = freqs[[context, symbol_index]] as u64;
        state = frequency * (state >> RANS_SCALE) + slot as u64 - cumulative[context][symbol_index];
        while state < RANS_L {
            if data_index >= data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "truncated rANS stream",
                ));
            }
            state = (state << 8) | data[data_index] as u64;
            data_index += 1;
        }
        *value = symbol_index as i64 + min_symbol;
        previous = *value;
    }
    Ok(output.into_pyarray_bound(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rans_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rans_decode, m)?)?;
    m.add_function(wrap_pyfunction!(ctx_rans_encode, m)?)?;
    m.add_function(wrap_pyfunction!(ctx_rans_decode, m)?)?;
    Ok(())
}
