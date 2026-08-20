//! `PyO3` bindings that expose `pybamm-core` to Python as `pybamm.rust._core`.
//!
//! Python builds a model by allocating nodes into an `ExprGraph`, then asks for
//! the artifact it needs: a `CompiledFunction` or `CompiledFunctionGroup` for
//! plain evaluation, a `CompiledJacobian` for derivatives, a `CompiledModel`
//! for the DAE the IDAKLU bridge drives, or a `PreparedSolver` to integrate in
//! Rust. Each names the core type it is the Python face of; core's per-solve
//! `ModelEvaluator` is reached through `CompiledModel.evaluator_pool`.
//! Each artifact is prepared once and owns pooled scratch, so a call reuses
//! that scratch rather than re-deriving it; the packed parameter vector and the
//! returned array are still allocated per call.
//!
//! Bindings own the boundary checks: array lengths and index ranges are validated
//! here so core code can assume them, and a core `CoreError` is mapped to
//! `ValueError` or `RuntimeError` rather than escaping as a panic.

use pyo3::prelude::*;

mod errors;
mod evaluator_pool;
mod expr;
mod function;
mod group;
mod jacobian;
mod model;
#[cfg(feature = "diffsol")]
mod pool;
mod scratch;
mod signature;
#[cfg(feature = "diffsol")]
mod solver;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<expr::ExprGraph>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<function::CompiledFunction>()?;
    m.add_class::<group::CompiledFunctionGroup>()?;
    m.add_class::<jacobian::CompiledJacobian>()?;
    m.add_class::<model::CompiledModel>()?;
    m.add_class::<evaluator_pool::EvaluatorPool>()?;

    #[cfg(feature = "diffsol")]
    {
        m.add_class::<solver::PreparedSolver>()?;
        m.add_class::<solver::SolverStatistics>()?;
        m.add_class::<solver::SolveOutcome>()?;
        m.add_function(wrap_pyfunction!(solver::default_solver_options, m)?)?;
        m.add_function(wrap_pyfunction!(pool::_pool_ids, m)?)?;
    }

    Ok(())
}
