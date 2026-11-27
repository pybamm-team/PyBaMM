//! `PyO3` bindings that expose `pybamm-core` to Python as `pybamm.rust._core`.
//!
//! Bindings own the boundary checks, so core code can assume valid lengths and
//! indices; classes carry `module = "pybamm.rust"` so pickle can import them.

use pyo3::prelude::*;

mod arrays;
mod compiled_function;
mod errors;
mod evaluator_capsule;
mod expr;
mod function_metadata;
mod jacobian;
mod model;
#[cfg(feature = "diffsol")]
mod rayon_pool;
mod scratch_pool;
#[cfg(feature = "diffsol")]
mod solver;

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<expr::ExprGraph>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<compiled_function::CompiledFunction>()?;
    m.add_class::<jacobian::CompiledJacobian>()?;
    m.add_class::<model::CompiledModel>()?;

    #[cfg(feature = "diffsol")]
    {
        m.add_class::<solver::PreparedSolver>()?;
        m.add_class::<solver::SolverStatistics>()?;
        m.add_class::<solver::SolveOutcome>()?;
        m.add_function(wrap_pyfunction!(solver::default_solver_options, m)?)?;
        m.add_function(wrap_pyfunction!(rayon_pool::pool_ids, m)?)?;
    }

    Ok(())
}
