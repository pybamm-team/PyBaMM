//! The equation bundle diffsol solves.
//!
//! Operator views are minted per call, since diffsol's operator traits borrow per call.

use std::sync::Arc;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    FaerContext, OdeEquations, OdeEquationsImplicitSens, OdeEquationsRef, Op, UnitCallable,
};

use super::init::InitOp;
use super::mass::MassOp;
use super::observable::ObservableOp;
use super::rhs::RhsOp;
use super::{FaerSparsity, lane, lane_mut};
use crate::model::ModelEvaluator;
use crate::params::ParamBinding;

/// Local implementation of diffsol's `OdeEquations` trait.
///
/// Views borrow from here rather than owning, so minting one per callback is cheap.
pub struct Equations {
    /// The model and this solve's workspace, which the operator views borrow.
    pub(crate) evaluator: ModelEvaluator,
    pub(crate) params: Vec<f64>,
    /// The packed inputs, or one differentiated parameter per entry on the sens path.
    pub(crate) binding: ParamBinding,
    pub(crate) y0: Vec<f64>,
    /// `dy0/dp`, column-major `n_states x binding.len()`; empty means zero.
    pub(crate) y0_sens: Vec<f64>,
    pub(crate) jac_sparsity: Arc<FaerSparsity>,
    pub(crate) mass_sparsity: Arc<FaerSparsity>,
    /// M's values in `mass_sparsity` order.
    pub(crate) mass_csc_values: Arc<[f64]>,
    pub(crate) context: FaerContext,
}

impl std::fmt::Debug for Equations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Equations")
            .field("n_states", &self.evaluator.compiled.n_states())
            .field("nparams", &self.binding.len())
            .finish_non_exhaustive()
    }
}

impl Op for Equations {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.evaluator.compiled.n_states()
    }
    fn nout(&self) -> usize {
        self.evaluator.compiled.n_states()
    }
    fn nparams(&self) -> usize {
        self.binding.len()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl<'a> OdeEquationsRef<'a> for Equations {
    type Rhs = RhsOp<'a>;
    type Mass = MassOp<'a>;
    type Root = ObservableOp<'a>;
    type Init = InitOp<'a>;
    /// Never constructed: the solve loop evaluates output variables itself.
    type Out = UnitCallable<FaerSparseMat<f64>>;
    /// Never constructed: `PyBaMM` events terminate the solve rather than reset it.
    type Reset = UnitCallable<FaerSparseMat<f64>>;
}

impl OdeEquations for Equations {
    fn rhs(&self) -> RhsOp<'_> {
        RhsOp {
            compiled: &self.evaluator.compiled,
            ws: &self.evaluator.workspace,
            inputs: &self.params,
            binding: &self.binding,
            jac_sparsity: &self.jac_sparsity,
            context: self.context,
        }
    }

    fn mass(&self) -> Option<MassOp<'_>> {
        Some(MassOp {
            compiled: &self.evaluator.compiled,
            sparsity: &self.mass_sparsity,
            csc_values: &self.mass_csc_values,
            context: self.context,
        })
    }

    fn root(&self) -> Option<ObservableOp<'_>> {
        (self.evaluator.compiled.total_event_len() > 0).then(|| ObservableOp {
            compiled: &self.evaluator.compiled,
            ws: &self.evaluator.workspace,
            inputs: &self.params,
            binding: &self.binding,
            context: self.context,
        })
    }

    fn out(&self) -> Option<UnitCallable<FaerSparseMat<f64>>> {
        None
    }

    fn init(&self) -> InitOp<'_> {
        InitOp {
            y0: &self.y0,
            y0_sens: &self.y0_sens,
            n_sens_params: self.binding.len(),
            context: self.context,
        }
    }

    /// Splice `p` into the packed inputs, leaving those it does not cover unchanged.
    fn set_params(&mut self, p: &FaerVec<f64>) {
        self.binding.splice_into(&mut self.params, lane(p));
    }

    /// Read `p` back out of the packed input vector.
    fn get_params(&self, p: &mut FaerVec<f64>) {
        self.binding.gather_from(&self.params, lane_mut(p));
    }
}

// Compile-time check that `Equations` meets the `OdeEquationsImplicitSens` bounds.
const _: fn() = || {
    const fn assert_implicit_sens<T: OdeEquationsImplicitSens>() {}
    assert_implicit_sens::<Equations>();
};
