//! Observable operator: the model's events as diffsol's `root` function.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op};

use super::{lane, lane_mut};

use crate::model::{CompiledModel, Workspace};
use crate::observable::ObservableKind;
use crate::params::ParamBinding;

/// The concatenated events as `H(t, y; p)`.
pub struct ObservableOp<'a> {
    pub compiled: &'a CompiledModel,
    pub ws: &'a RefCell<Workspace>,
    pub inputs: &'a [f64],
    /// What the solve's parameter vector stands for.
    pub binding: &'a ParamBinding,
    pub context: FaerContext,
}

/// The only family a solve hands diffsol.
const KIND: ObservableKind = ObservableKind::Events;

impl std::fmt::Debug for ObservableOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservableOp")
            .field("n_out", &self.nout())
            .finish_non_exhaustive()
    }
}

impl Op for ObservableOp<'_> {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.compiled.n_states()
    }
    /// The events' concatenated length.
    fn nout(&self) -> usize {
        self.compiled.observables(KIND).total_len()
    }
    fn nparams(&self) -> usize {
        self.binding.len()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl NonLinearOp for ObservableOp<'_> {
    fn call_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled
            .eval_observables(&mut ws, KIND, t, lane(x), self.inputs, lane_mut(y));
    }
}

impl NonLinearOpJacobian for ObservableOp<'_> {
    /// dH/dy · v.
    fn jac_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.observable_jac_action(
            &mut ws,
            KIND,
            t,
            lane(x),
            self.inputs,
            lane(v),
            lane_mut(y),
        );
    }
}

impl NonLinearOpSens for ObservableOp<'_> {
    /// dH/dp · v.
    fn sens_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.observable_sens_action(
            &mut ws,
            KIND,
            t,
            lane(x),
            self.inputs,
            lane(v),
            lane_mut(y),
        );
    }
}
