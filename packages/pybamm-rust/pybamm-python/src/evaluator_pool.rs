//! N independent [`ModelEvaluator`]s over one shared compiled tape, giving each
//! `IDAKLUSolverGroup` thread its own scratch without N lowerings.

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;

use pybamm_core::{CompiledModel, ModelEvaluator};

// `as_ptr` moves an evaluator onto a C++ solver thread, so it must stay `Send`.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ModelEvaluator>();
};

/// One pooled evaluator; Rust never reads it again once its address is handed out.
struct Slot {
    /// Boxed so the handed-out address stays fixed for the pool's life.
    evaluator: Box<ModelEvaluator>,
    handed_out: bool,
}

/// N [`ModelEvaluator`]s over one shared compiled tape.
///
/// C++ writes through each handed-out address, so the pool must outlive that use.
#[pyclass(frozen, module = "pybamm.rust")]
pub struct EvaluatorPool {
    slots: Box<[Mutex<Slot>]>,
}

impl std::fmt::Debug for EvaluatorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvaluatorPool")
            .field("len", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl EvaluatorPool {
    /// Build a pool of `n` evaluators sharing one compiled model.
    pub(crate) fn from_compiled(compiled: &Arc<CompiledModel>, n: usize) -> Self {
        Self {
            slots: (0..n)
                .map(|_| {
                    Mutex::new(Slot {
                        evaluator: Box::new(ModelEvaluator::from_compiled(Arc::clone(compiled))),
                        handed_out: false,
                    })
                })
                .collect(),
        }
    }
}

#[pymethods]
impl EvaluatorPool {
    /// Address of evaluator `index` for a C++ solver; each is handed out once.
    ///
    /// The caller must keep this pool alive while the address is in use.
    fn as_ptr(&self, index: usize) -> PyResult<usize> {
        let slot = self.slots.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "evaluator index {index} is out of range for a pool of {}",
                self.slots.len()
            ))
        })?;
        // Nothing can panic under the lock, so a poisoned slot is still consistent.
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if std::mem::replace(&mut slot.handed_out, true) {
            return Err(PyRuntimeError::new_err(format!(
                "evaluator {index} was already handed to a solver; each address \
                 is given out once — build a new pool for a new solver group"
            )));
        }
        // Not `as usize`: C++ casts the integer back to a pointer and writes through it.
        Ok(std::ptr::from_mut(&mut *slot.evaluator).expose_provenance())
    }

    fn __len__(&self) -> usize {
        self.slots.len()
    }

    fn __repr__(&self) -> String {
        format!("EvaluatorPool(len={})", self.slots.len())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use pybamm_core::ffi::{SUCCESS, pybamm_rust_eval_rhs};
    use pybamm_core::{Arena, CompiledModelOptions, CsrData, Node, PackedWidth, Shape};

    use super::*;

    /// A pool over `f(y) = y` on two states with an identity mass matrix.
    fn identity_pool(n: usize) -> EvaluatorPool {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let mass = CsrData::try_new(vec![0, 1, 2], vec![0, 1], vec![1.0; 2], Shape::matrix(2, 2))
            .expect("identity CSR is valid");
        let compiled = CompiledModel::new(
            &arena,
            y,
            mass,
            2,
            PackedWidth::new(0),
            CompiledModelOptions::default(),
        )
        .expect("a valid fixture");
        EvaluatorPool::from_compiled(&Arc::new(compiled), n)
    }

    #[test]
    fn each_live_evaluator_is_handed_out_once() {
        let pool = identity_pool(3);
        let addresses: Vec<usize> = (0..3).filter_map(|i| pool.as_ptr(i).ok()).collect();
        let distinct: std::collections::HashSet<_> = addresses.iter().collect();
        assert_eq!((addresses.len(), distinct.len()), (3, 3));
        assert!(pool.as_ptr(1).is_err(), "a second hand-out must be refused");
        assert!(
            pool.as_ptr(3).is_err(),
            "an out-of-range index must be refused"
        );

        let y = [1.0, 2.0];
        for address in addresses {
            let mut rhs = [0.0; 2];
            // SAFETY: the pool is alive and this loop is the address's only user, as C++ is.
            #[allow(unsafe_code)]
            let status = unsafe {
                pybamm_rust_eval_rhs(
                    0.0,
                    y.as_ptr(),
                    std::ptr::null(),
                    rhs.as_mut_ptr(),
                    std::ptr::with_exposed_provenance_mut::<c_void>(address),
                )
            };
            assert_eq!(status, SUCCESS);
            assert_eq!(rhs.map(f64::to_bits), [1.0_f64, 2.0].map(f64::to_bits));
        }
    }
}
