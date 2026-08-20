//! Conversion from `pybamm_core` errors to Python exceptions.

use pybamm_core::CoreError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Map a core error to a Python exception. Caller-argument and
/// invalid-data problems (empty `t_eval`, mismatched `y0`/inputs/atol, or a
/// malformed matrix/interpolant crossing the boundary) become `ValueError`;
/// an integration failure inside diffsol becomes `RuntimeError`.
pub fn core_err_to_py(err: CoreError) -> PyErr {
    match err {
        #[cfg(feature = "diffsol")]
        CoreError::Diffsol(source) => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("diffsol error: {source}"))
        },
        other => PyValueError::new_err(other.to_string()),
    }
}
