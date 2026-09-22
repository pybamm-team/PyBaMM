//! Conversion from `pybamm_core` errors to Python exceptions.

use pybamm_core::CoreError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Map a core error to a Python `ValueError`; every core error is a caller-argument or
/// invalid-data problem.
// By value, because every caller hands it over through `map_err`.
#[allow(clippy::needless_pass_by_value)]
pub fn core_err_to_py(err: CoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}
