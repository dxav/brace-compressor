//! Python module entry point for the optional BRACE Rust extension.

mod rans;
mod scan;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rans::register(m)?;
    scan::register(m)?;
    Ok(())
}
