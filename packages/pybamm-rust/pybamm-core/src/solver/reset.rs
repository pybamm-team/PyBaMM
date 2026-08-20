//! Placeholder reset operator.
//!
//! `PyBaMM` events terminate the solve rather than triggering a state reset, so
//! `Equations::reset()` always returns `None` and `ResetOp` is never
//! constructed. It exists solely to satisfy the `Reset` associated-type bounds
//! of `OdeEquationsImplicitSens`.

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op, VectorHost};

/// Never-constructed reset operator; satisfies the diffsol `Reset` bounds.
pub struct ResetOp {
    /// Number of state variables.
    pub n_states: usize,
    /// Number of sensitivity columns.
    pub n_sens_params: usize,
    /// Faer execution context.
    pub context: FaerContext,
}

impl std::fmt::Debug for ResetOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResetOp")
            .field("n_states", &self.n_states)
            .field("n_sens_params", &self.n_sens_params)
            .finish_non_exhaustive()
    }
}

impl Op for ResetOp {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn context(&self) -> &Self::C {
        &self.context
    }

    fn nstates(&self) -> usize {
        self.n_states
    }

    fn nout(&self) -> usize {
        self.n_states
    }

    fn nparams(&self) -> usize {
        self.n_sens_params
    }
}

impl NonLinearOp for ResetOp {
    /// Identity reset: new state equals old state.
    fn call_inplace(&self, x: &FaerVec<f64>, _t: f64, y: &mut FaerVec<f64>) {
        y.as_mut_slice().copy_from_slice(x.as_slice());
    }
}

impl NonLinearOpJacobian for ResetOp {
    /// Identity Jacobian-vector product.
    fn jac_mul_inplace(&self, _x: &FaerVec<f64>, _t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        y.as_mut_slice().copy_from_slice(v.as_slice());
    }
}

impl NonLinearOpSens for ResetOp {
    /// Zero parameter sensitivity.
    fn sens_mul_inplace(
        &self,
        _x: &FaerVec<f64>,
        _t: f64,
        _v: &FaerVec<f64>,
        y: &mut FaerVec<f64>,
    ) {
        y.as_mut_slice().fill(0.0);
    }
}
