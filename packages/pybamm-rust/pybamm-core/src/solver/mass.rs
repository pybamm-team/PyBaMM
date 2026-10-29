//! Mass matrix operator; `M` is constant and arrives from Python in CSR.
//!
//! A zero *column* `i` makes state `i` algebraic. diffsol's init partition tests the diagonal
//! instead, which disagrees only on a zero diagonal beside an off-diagonal non-zero.

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, LinearOp, Matrix, Op, Vector};

use super::{FaerSparsity, values_mut};
use crate::model::CompiledModel;

/// Mass operator `M`, constant in `t` and `p`.
pub struct MassOp<'a> {
    pub compiled: &'a CompiledModel,
    pub sparsity: &'a FaerSparsity,
    /// M's values in the CSC order of `sparsity`.
    pub csc_values: &'a [f64],
    pub context: FaerContext,
}

impl std::fmt::Debug for MassOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MassOp")
            .field("n_states", &self.compiled.n_states())
            .finish_non_exhaustive()
    }
}

impl Op for MassOp<'_> {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.compiled.n_states()
    }
    fn nout(&self) -> usize {
        self.compiled.n_states()
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl LinearOp for MassOp<'_> {
    /// Compute `y = M @ x + beta * y` for each lane of `x` and `y`.
    fn gemv_inplace(&self, x: &FaerVec<f64>, _t: f64, beta: f64, y: &mut FaerVec<f64>) {
        FaerVec::for_each_batch_mut_host([y], [x], |[y], [x], _| {
            self.compiled.mass_gemv(x, beta, y);
        });
    }

    /// Copy M's precomputed values into `y`, which uses this operator's `sparsity`.
    fn matrix_inplace(&self, _t: f64, y: &mut FaerSparseMat<f64>) {
        values_mut(y).copy_from_slice(self.csc_values);
    }

    fn sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
        Some(self.sparsity.clone())
    }
}
