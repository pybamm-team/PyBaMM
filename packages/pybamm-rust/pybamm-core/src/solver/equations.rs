//! The equation bundle diffsol solves.
//!
//! [`Equations`] owns what a solve shares: the compiled model, workspace, this solve's
//! parameter values and its sensitivity-parameter indices. It mints a fresh
//! operator view whenever diffsol asks for one, because the operator traits
//! borrow for the length of a call rather than for the solve.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    FaerContext, OdeEquations, OdeEquationsImplicitSens, OdeEquationsRef, Op, VectorHost,
};

use super::FaerSparsity;
use super::init::InitOp;
use super::mass::MassOp;
use super::observable::ObservableOp;
use super::reset::ResetOp;
use super::rhs::RhsOp;
use crate::model::{CompiledModel, Workspace};
use crate::observable::ObservableKind;

/// Local ODE-equations container implementing diffsol's public `OdeEquations`
/// trait.
///
/// diffsol mints an operator view per callback invocation — `eqn.rhs()` on every
/// residual call, and again for every `nstates()` — so the views borrow from
/// here instead of owning: a mint is a few pointer copies, and the compiler,
/// not a refcount, keeps this alive for as long as a view exists.
///
/// `Arc` marks the state shared with the `PreparedSolver` across solves, which
/// is `Send + Sync`; everything built per solve is owned outright.
pub struct Equations {
    pub(crate) compiled: Arc<CompiledModel>,
    /// Shared with the caller, which evaluates output tapes against the same
    /// scratch between steps on the output-variable path.
    pub(crate) ws: Rc<RefCell<Workspace>>,
    pub(crate) params: Vec<f64>,
    /// Global parameter index of each sensitivity column; the identity on the
    /// ordinary no-sensitivity path.
    pub(crate) sens_params: Arc<[usize]>,
    pub(crate) y0: Vec<f64>,
    /// `dy0/dp`, column-major `n_states x sens_params.len()`; empty means zero.
    pub(crate) y0_sens: Vec<f64>,
    pub(crate) jac_sparsity: Arc<FaerSparsity>,
    pub(crate) mass_sparsity: Arc<FaerSparsity>,
    /// M's values in `mass_sparsity` order, shared with every `MassOp` view.
    pub(crate) mass_csc_values: Arc<[f64]>,
    pub(crate) context: FaerContext,
    pub(crate) n_states: usize,
    pub(crate) n_event_outputs: usize,
    pub(crate) n_outputs: usize,
    pub(crate) with_output: bool,
}

impl std::fmt::Debug for Equations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Equations")
            .field("n_states", &self.n_states)
            // sens_params is the identity 0..n_params on the plain path, so this
            // is nparams(), not a sensitivity-subset count, until sens is active.
            .field("nparams", &self.sens_params.len())
            .field("n_event_outputs", &self.n_event_outputs)
            .field("n_outputs", &self.n_outputs)
            .field("with_output", &self.with_output)
            .finish_non_exhaustive()
    }
}

impl Equations {
    /// Mint an operator view onto one observable family.
    fn observable_op(&self, kind: ObservableKind) -> ObservableOp<'_> {
        ObservableOp {
            compiled: &self.compiled,
            ws: &self.ws,
            inputs: &self.params,
            sens_params: &self.sens_params,
            n_states: self.n_states,
            kind,
            n_out: match kind {
                ObservableKind::Outputs => self.n_outputs,
                ObservableKind::Events => self.n_event_outputs,
            },
            context: self.context,
        }
    }
}

impl Op for Equations {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        self.n_states
    }
    fn nout(&self) -> usize {
        if self.with_output && self.n_outputs > 0 {
            self.n_outputs
        } else {
            self.n_states
        }
    }
    fn nparams(&self) -> usize {
        self.sens_params.len()
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
    type Out = ObservableOp<'a>;
    /// Never constructed: `reset()` always returns `None` for `PyBaMM`
    /// (events terminate, never reset). `ResetOp` satisfies the bound.
    type Reset = ResetOp;
}

impl OdeEquations for Equations {
    fn rhs(&self) -> RhsOp<'_> {
        RhsOp {
            compiled: &self.compiled,
            ws: &self.ws,
            inputs: &self.params,
            sens_params: &self.sens_params,
            jac_sparsity: &self.jac_sparsity,
            n_states: self.n_states,
            context: self.context,
        }
    }

    fn mass(&self) -> Option<MassOp<'_>> {
        Some(MassOp {
            compiled: &self.compiled,
            ws: &self.ws,
            sparsity: &self.mass_sparsity,
            csc_values: &self.mass_csc_values,
            n_states: self.n_states,
            context: self.context,
        })
    }

    fn root(&self) -> Option<ObservableOp<'_>> {
        (self.n_event_outputs > 0).then(|| self.observable_op(ObservableKind::Events))
    }

    fn out(&self) -> Option<ObservableOp<'_>> {
        (self.with_output && self.n_outputs > 0)
            .then(|| self.observable_op(ObservableKind::Outputs))
    }

    fn init(&self) -> InitOp<'_> {
        InitOp {
            y0: &self.y0,
            y0_sens: &self.y0_sens,
            n_states: self.n_states,
            n_sens_params: self.sens_params.len(),
            context: self.context,
        }
    }

    /// Splice the `k` sensitivity values into the full input vector at their
    /// global indices, leaving the carried-but-not-differentiated inputs alone.
    fn set_params(&mut self, p: &FaerVec<f64>) {
        for (&global, &v) in self.sens_params.iter().zip(p.as_slice()) {
            self.params[global] = v;
        }
    }

    /// Gather the `k` sensitivity entries out of the full input vector. diffsol
    /// sizes `p` by `nparams()`, so this cannot be a wholesale copy.
    fn get_params(&self, p: &mut FaerVec<f64>) {
        for (dst, &global) in p.as_mut_slice().iter_mut().zip(self.sens_params.iter()) {
            *dst = self.params[global];
        }
    }
}

// Pins that `Equations` satisfies the forward-sensitivity bounds, which
// diffsol's blanket `impl OdeEquationsImplicitSens` needs.
const _: fn() = || {
    const fn assert_implicit_sens<T: OdeEquationsImplicitSens>() {}
    assert_implicit_sens::<Equations>();
};
