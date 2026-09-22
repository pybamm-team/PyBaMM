//! `PyO3` bindings that expose `pybamm-core` to Python as `pybamm.rust._core`.
//!
//! Bindings own the boundary checks, so core code can assume valid lengths and
//! indices; classes carry `module = "pybamm.rust"` so pickle can import them.

use pyo3::prelude::*;

mod compiled_function;
mod errors;
mod expr;
mod function_metadata;
mod jacobian;
mod scratch_pool;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<expr::ExprGraph>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<compiled_function::CompiledFunction>()?;
    m.add_class::<jacobian::CompiledJacobian>()?;

    Ok(())
}
