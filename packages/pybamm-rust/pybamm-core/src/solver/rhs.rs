//! Right-hand-side operator: `f(t, y; p)`, its sparse `df/dy`, and `df/dp`.
//!
//! `df/dp` columns are numbered by position in the parameter subset, not registration index.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    Context, FaerContext, Matrix, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op, Vector,
};

use super::{FaerSparsity, dense_faer_sparsity, lane, lane_mut, lanes, lanes_mut, values_mut};
use crate::model::{CompiledModel, Workspace};
use crate::params::ParamBinding;

/// RHS operator `f(t, y; p)`.
pub struct RhsOp<'a> {
    pub compiled: &'a CompiledModel,
    pub ws: &'a RefCell<Workspace>,
    pub inputs: &'a [f64],
    /// What the solve's parameter vector stands for.
    pub binding: &'a ParamBinding,
    pub jac_sparsity: &'a FaerSparsity,
    pub context: FaerContext,
}

impl std::fmt::Debug for RhsOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhsOp")
            .field("n_states", &self.compiled.n_states())
            .field("nparams", &self.binding.len())
            .finish_non_exhaustive()
    }
}

impl Op for RhsOp<'_> {
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
        self.binding.len()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl NonLinearOp for RhsOp<'_> {
    fn call_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .eval_rhs(&mut ws, t, lane(x), self.inputs, lane_mut(y));
    }
}

impl NonLinearOpJacobian for RhsOp<'_> {
    /// Write `df/dy · v` into `y` for each lane of `v`, all at state `x`.
    fn jac_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        assert_eq!(
            v.context().nbatch(),
            y.context().nbatch(),
            "jac_mul direction and output lanes differ"
        );
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .jac_action_at(&mut ws, t, lane(x), self.inputs)
            .apply_lanes(lanes(v), lanes_mut(y));
    }

    fn jacobian_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerSparseMat<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.assemble_jacobian_csc_no_mass(
            &mut ws,
            t,
            lane(x),
            self.inputs,
            values_mut(y),
        );
    }

    fn jacobian_sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
        Some(self.jac_sparsity.clone())
    }
}

impl NonLinearOpSens for RhsOp<'_> {
    fn sens_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .sens_action(&mut ws, t, lane(x), self.inputs, lane(v), lane_mut(y));
    }

    /// Batched `df/dp`: one primal pass, then a tangent-only sweep per column.
    ///
    /// `y` must use the dense pattern of [`Self::sens_sparsity`].
    fn sens_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerSparseMat<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .eval_sens_all(&mut ws, t, lane(x), self.inputs, values_mut(y));
    }

    /// Dense pattern for `df/dp`: every state may depend on every parameter.
    fn sens_sparsity(&self) -> Option<FaerSparsity> {
        Some(dense_faer_sparsity(
            self.compiled.n_states(),
            self.binding.len(),
        ))
    }
}
