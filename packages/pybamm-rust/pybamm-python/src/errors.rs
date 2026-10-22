//! Conversion from `pybamm_core` errors to Python exceptions.

use pybamm_core::CoreError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Map a core error to a Python exception: caller-argument and invalid-data
/// problems (a malformed matrix or interpolant crossing the boundary) become
/// `ValueError`.
// By value, because every caller hands it over through `map_err`.
#[allow(clippy::needless_pass_by_value)]
pub fn core_err_to_py(err: CoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
