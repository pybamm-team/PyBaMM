//! Conversion from `pybamm_core` errors to Python exceptions.

use pybamm_core::{CoreError, GraphError};
use pyo3::PyErr;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};

/// Map a refused graph to `NotImplementedError` when unsupported, else `ValueError`.
pub fn graph_err_to_py(err: GraphError) -> PyErr {
    match err {
        GraphError::Unsupported(msg) => PyNotImplementedError::new_err(format!(
            "Rust conversion does not support this expression: {msg}. Use a \
             CasADi-backed model (set `model.convert_to_format = 'casadi'`)."
        )),
        GraphError::Invalid(msg) => {
            PyValueError::new_err(format!("Invalid Rust expression graph: {msg}"))
        },
    }
}

/// Map a refused graph as [`graph_err_to_py`] does, and any other core error to a
/// `ValueError`: a caller-argument or invalid-data problem.
pub fn core_err_to_py(err: CoreError) -> PyErr {
    match err {
        CoreError::Graph(graph) => graph_err_to_py(graph),
        other => PyValueError::new_err(other.to_string()),
    }
}
