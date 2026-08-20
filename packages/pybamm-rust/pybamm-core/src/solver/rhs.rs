//! Right-hand-side operator.
//!
//! Supplies `f(t, y; p)`, its sparse `df/dy` assembled through the compiled
//! coloring, and the `df/dp` columns forward sensitivities need. Those columns
//! are numbered by position within the solve's parameter subset; `sens_params`
//! maps each back to its global parameter index.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    FaerContext, Matrix, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op, VectorHost,
};

use super::{FaerSparsity, dense_faer_sparsity};
use crate::model::{CompiledModel, Workspace};

/// RHS operator f(t, y; p), borrowed from the [`Equations`](super::equations::Equations)
/// that owns the solve.
///
/// diffsol mints one of these per callback invocation, so it holds nothing of
/// its own: every field is a reference into the equations, making a mint a
/// handful of pointer copies.
pub struct RhsOp<'a> {
    pub compiled: &'a CompiledModel,
    pub ws: &'a RefCell<Workspace>,
    pub inputs: &'a [f64],
    /// Global parameter index of each sensitivity column.
    pub sens_params: &'a [usize],
    pub jac_sparsity: &'a FaerSparsity,
    pub n_states: usize,
    pub context: FaerContext,
}

impl std::fmt::Debug for RhsOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhsOp")
            .field("n_states", &self.n_states)
            .field("n_sens_params", &self.sens_params.len())
            .finish_non_exhaustive()
    }
}

impl Op for RhsOp<'_> {
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
        self.sens_params.len()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl NonLinearOp for RhsOp<'_> {
    fn call_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .eval_rhs(&mut ws, t, x.as_slice(), self.inputs, y.as_mut_slice());
    }
}

impl NonLinearOpJacobian for RhsOp<'_> {
    fn jac_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.jac_action(
            &mut ws,
            t,
            x.as_slice(),
            self.inputs,
            v.as_slice(),
            y.as_mut_slice(),
        );
    }

    fn jacobian_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerSparseMat<f64>) {
        let mut ws = self.ws.borrow_mut();
        let values = y.inner_mut().val_mut();
        self.compiled
            .assemble_jacobian_csc_no_mass(&mut ws, t, x.as_slice(), self.inputs, values);
    }

    /// Deep-copies the borrowed pattern: diffsol asks once per problem build,
    /// so the owned copy its signature requires is paid there rather than on
    /// every operator mint.
    fn jacobian_sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
        Some(self.jac_sparsity.clone())
    }
}

impl NonLinearOpSens for RhsOp<'_> {
    fn sens_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.sens_action(
            &mut ws,
            t,
            x.as_slice(),
            self.inputs,
            self.sens_params,
            v.as_slice(),
            y.as_mut_slice(),
        );
    }

    /// Batched df/dp: one primal pass per (t, y), then a tangent-only sweep per
    /// column, instead of diffsol's default per-column primal recompute.
    ///
    /// Writes each column straight into `y`'s value array. diffsol builds `y`
    /// from [`Self::sens_sparsity`], whose dense CSC pattern puts column `dst`
    /// at `dst * n_states`, so no intermediate column vector is needed.
    fn sens_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerSparseMat<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .sens_primal_pass(&mut ws, t, x.as_slice(), self.inputs);
        let values = y.inner_mut().val_mut();
        debug_assert_eq!(
            values.len(),
            self.n_states * self.sens_params.len(),
            "df/dp matrix was not built from sens_sparsity",
        );
        for (dst, &param) in self.sens_params.iter().enumerate() {
            let column = &mut values[dst * self.n_states..(dst + 1) * self.n_states];
            self.compiled.sens_tangent_column(&mut ws, param, column);
        }
    }

    /// Dense (all-entries) pattern for df/dp over the requested subset: every
    /// state may depend on every parameter.
    fn sens_sparsity(&self) -> Option<FaerSparsity> {
        Some(dense_faer_sparsity(self.n_states, self.sens_params.len()))
    }
}
