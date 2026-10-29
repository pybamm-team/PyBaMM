//! Conversion from `pybamm_core` errors to Python exceptions.

use pybamm_core::CoreError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Map a core error to `RuntimeError` for integration failures, else `ValueError`.
pub fn core_err_to_py(err: CoreError) -> PyErr {
    match err {
        #[cfg(feature = "diffsol")]
        CoreError::Diffsol(source) => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("diffsol error: {source}"))
        },
        retry @ CoreError::SensRetryFailed { .. } => {
            pyo3::exceptions::PyRuntimeError::new_err(retry.to_string())
        },
        other => PyValueError::new_err(other.to_string()),
    }
}
