//! `PyO3` bindings that expose `pybamm-core` to Python as `pybamm.rust._core`.
//!
//! Bindings own the boundary checks, so core code can assume valid lengths and
//! indices; classes carry `module = "pybamm.rust"` so pickle can import them.

use pyo3::prelude::*;

mod compiled_function;
mod errors;
mod evaluator_pool;
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
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<expr::ExprGraph>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<compiled_function::CompiledFunction>()?;
    m.add_class::<jacobian::CompiledJacobian>()?;
    m.add_class::<model::CompiledModel>()?;
    m.add_class::<evaluator_pool::EvaluatorPool>()?;

    #[cfg(feature = "diffsol")]
    {
        m.add_class::<solver::PreparedSolver>()?;
        m.add_class::<solver::SolverStatistics>()?;
        m.add_class::<solver::SolveOutcome>()?;
        m.add_function(wrap_pyfunction!(solver::default_solver_options, m)?)?;
        m.add_function(wrap_pyfunction!(rayon_pool::_pool_ids, m)?)?;
    }

    Ok(())
}
