//! Observable operator: one family of observables as a diffsol operator.
//!
//! diffsol wants both of `PyBaMM`'s observable families as a vector-valued
//! function of `(t, y; p)` with a jacobian and a sens action: output variables
//! as its `out`, events as its `root`. That is the same operator over a
//! different [`ObservableKind`], so it is written once and mounted twice.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op, VectorHost};

use crate::model::{CompiledModel, Workspace};
use crate::observable::ObservableKind;

/// One observable family as `H(t, y; p)`.
///
/// Acting on an event root is the caller's job, not the operator's, so events
/// need nothing here that outputs do not.
pub struct ObservableOp<'a> {
    pub compiled: &'a CompiledModel,
    pub ws: &'a RefCell<Workspace>,
    pub inputs: &'a [f64],
    /// Global parameter index of each sensitivity column.
    pub sens_params: &'a [usize],
    pub n_states: usize,
    pub kind: ObservableKind,
    /// The family's concatenated length, which diffsol asks for per callback.
    pub n_out: usize,
    pub context: FaerContext,
}

impl std::fmt::Debug for ObservableOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservableOp")
            .field("kind", &self.kind)
            .field("n_out", &self.n_out)
            .finish_non_exhaustive()
    }
}

impl Op for ObservableOp<'_> {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.n_states
    }
    fn nout(&self) -> usize {
        self.n_out
    }
    fn nparams(&self) -> usize {
        self.sens_params.len()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl NonLinearOp for ObservableOp<'_> {
    fn call_inplace(&self, x: &FaerVec<f64>, t: f64, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.eval_observables(
            &mut ws,
            self.kind,
            t,
            x.as_slice(),
            self.inputs,
            y.as_mut_slice(),
        );
    }
}

impl NonLinearOpJacobian for ObservableOp<'_> {
    /// dH/dy · v.
    fn jac_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.observable_jac_action(
            &mut ws,
            self.kind,
            t,
            x.as_slice(),
            self.inputs,
            v.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl NonLinearOpSens for ObservableOp<'_> {
    /// dH/dp · v.
    fn sens_mul_inplace(&self, x: &FaerVec<f64>, t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let mut ws = self.ws.borrow_mut();
        self.compiled.observable_sens_action(
            &mut ws,
            self.kind,
            t,
            x.as_slice(),
            self.inputs,
            self.sens_params,
            v.as_slice(),
            y.as_mut_slice(),
        );
    }
}
