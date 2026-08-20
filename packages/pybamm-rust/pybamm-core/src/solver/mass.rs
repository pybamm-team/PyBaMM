//! Mass matrix operator.
//!
//! `M` is constant and arrives from Python in CSR, so the operator only applies
//! it. A zero row is how an algebraic equation reaches the solver: it makes the
//! system a DAE in that row rather than an ODE.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, LinearOp, Matrix, Op, VectorHost};

use super::FaerSparsity;
use crate::model::{CompiledModel, Workspace};

/// Mass operator M (constant in t and p), borrowed from the
/// [`Equations`](super::equations::Equations) that owns the solve.
///
/// diffsol mints one of these per callback invocation, so every field is a
/// reference into the equations: a mint copies pointers, never data.
pub struct MassOp<'a> {
    pub compiled: &'a CompiledModel,
    pub ws: &'a RefCell<Workspace>,
    pub sparsity: &'a FaerSparsity,
    /// M's values in the CSC order of `sparsity` (see `csr_mass_to_faer_csc`),
    /// so `matrix_inplace` is a copy rather than diffsol's column probing.
    pub csc_values: &'a [f64],
    pub n_states: usize,
    pub context: FaerContext,
}

impl std::fmt::Debug for MassOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MassOp")
            .field("n_states", &self.n_states)
            .finish_non_exhaustive()
    }
}

impl Op for MassOp<'_> {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.n_states
    }
    fn nout(&self) -> usize {
        self.n_states
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl LinearOp for MassOp<'_> {
    /// y = M @ x + beta * y. `mass_action` computes M @ x with no beta, so we
    /// fold beta ourselves via the workspace `mv_buffer` when beta != 0.
    fn gemv_inplace(&self, x: &FaerVec<f64>, _t: f64, beta: f64, y: &mut FaerVec<f64>) {
        if beta == 0.0 {
            self.compiled.mass_action(x.as_slice(), y.as_mut_slice());
        } else {
            let mut ws = self.ws.borrow_mut();
            self.compiled.mass_action(x.as_slice(), &mut ws.mv_buffer);
            let ys = y.as_mut_slice();
            for (yi, &mvi) in ys.iter_mut().zip(ws.mv_buffer.iter()).take(self.n_states) {
                *yi = beta.mul_add(*yi, mvi);
            }
        }
    }

    /// M is constant: copy the precomputed CSC values into `y`, which diffsol
    /// allocated from this operator's `sparsity`. Overrides the trait default,
    /// which probes M one unit-vector gemv per column (O(n²) per Jacobian)
    fn matrix_inplace(&self, _t: f64, y: &mut FaerSparseMat<f64>) {
        y.inner_mut().val_mut().copy_from_slice(self.csc_values);
    }

    /// Deep-copies the borrowed pattern, which is what the signature requires.
    /// diffsol asks once per problem build, so the copy is paid there rather
    /// than on every operator mint.
    fn sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
        Some(self.sparsity.clone())
    }
}
