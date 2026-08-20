//! N independent [`ModelEvaluator`]s over one shared compiled tape.
//!
//! An `IDAKLUSolverGroup` runs one solver per OpenMP thread, and each solver
//! evaluates through the FFI with `&mut ModelEvaluator`. The tape is not what
//! stops that being concurrent — the scratch is, so the pool mints one
//! `Workspace` per solver against the one immutable `CompiledModel` behind the
//! `Arc`. N independent `CompiledModel`s would mean N lowerings and N tape
//! copies, which is the cost the `Arc` exists to avoid.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;

use std::sync::Arc;

use pybamm_core::{CompiledModel, ModelEvaluator};

// `as_ptr` moves exclusive access to one evaluator onto a C++ solver thread,
// so a thread-bound field added to `Workspace` must fail here, not data-race.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ModelEvaluator>();
};

/// N [`ModelEvaluator`]s over one shared compiled tape.
///
/// `UnsafeCell` because C++ writes through each address for as long as the
/// solver group lives, so a `*const` derived from a shared reference and cast
/// to `*mut` would be the wrong provenance under Stacked/Tree Borrows. The
/// `Vec` is filled once in [`from_compiled`](Self::from_compiled) and never
/// resized, so the addresses it hands out stay valid for the pool's life.
#[pyclass(module = "pybamm.rust")]
pub struct EvaluatorPool {
    evaluators: Vec<UnsafeCell<ModelEvaluator>>,
    /// Take-once flag per evaluator, set when `as_ptr` hands its address out.
    taken: Vec<AtomicBool>,
}

// SAFETY: the pool never reads or writes through a cell itself, and `as_ptr`
// hands each evaluator's address out at most once (`taken`), so every address
// has exactly one writer; the `CompiledModel` behind the `Arc` is immutable.
#[allow(unsafe_code)]
unsafe impl Sync for EvaluatorPool {}

impl std::fmt::Debug for EvaluatorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvaluatorPool")
            .field("len", &self.evaluators.len())
            .finish_non_exhaustive()
    }
}

impl EvaluatorPool {
    /// Build a pool of `n` evaluators over one shared compiled model, which
    /// allocates `n` scratches and no tapes.
    pub(crate) fn from_compiled(compiled: &Arc<CompiledModel>, n: usize) -> Self {
        Self {
            evaluators: (0..n)
                .map(|_| UnsafeCell::new(ModelEvaluator::from_compiled(Arc::clone(compiled))))
                .collect(),
            taken: (0..n).map(|_| AtomicBool::new(false)).collect(),
        }
    }
}

#[pymethods]
impl EvaluatorPool {
    /// Address of evaluator `index`, for the C++ solver that will drive it.
    ///
    /// Each address is handed out at most once — a second take raises
    /// `RuntimeError` — so no two solvers can be given the same evaluator;
    /// build a new pool per solver group rather than re-taking from one.
    /// The caller must keep this pool alive for as long as the address is used;
    /// `IDAKLUSolverGroup` does that by holding the pool itself.
    fn as_ptr(&self, index: usize) -> PyResult<usize> {
        let cell = self.evaluators.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "evaluator index {index} is out of range for a pool of {}",
                self.evaluators.len()
            ))
        })?;
        if self.taken[index].swap(true, Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err(format!(
                "evaluator {index} was already handed to a solver; each address \
                 is given out once — build a new pool for a new solver group"
            )));
        }
        // expose_provenance, not `as usize`: the integer crosses to C++ and is
        // cast straight back to a pointer that is then written through.
        Ok(cell.get().expose_provenance())
    }

    const fn __len__(&self) -> usize {
        self.evaluators.len()
    }

    fn __repr__(&self) -> String {
        format!("EvaluatorPool(len={})", self.evaluators.len())
    }
}
