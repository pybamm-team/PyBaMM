//! Models shared by the solver's test modules.

use crate::arena::Arena;
use crate::model::CompiledModel;
use crate::model::test_fixtures::{compile, compile_with_sens, identity_mass_matrix, input};
use crate::node::{CsrData, Node, Shape};
use crate::observable::ObservableKind;
use crate::params::ParamIdx;

/// `diag(1, 0)`: a differential state, then an algebraic one.
pub fn differential_then_algebraic_mass() -> CsrData {
    CsrData {
        indptr: vec![0, 1, 1],
        indices: vec![0],
        data: vec![1.0],
        shape: Shape::matrix(2, 2),
    }
}

/// `dy/dt = -y` on one state, with no inputs and so no sensitivity parameters.
pub fn decay() -> CompiledModel {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let neg = arena.alloc(Node::Scalar(-1.0));
    let rhs = arena.alloc(Node::Mul(neg, y));
    compile(&arena, rhs, identity_mass_matrix(1), 1, 0)
}

/// `dy/dt = -a*y`, output `2*y`, event `y - 0.4`, sensitivities wrt `a`.
///
/// From `y = 1` with `a = 1` the event fires at `t = ln 2.5`.
pub fn decay_with_output_and_event() -> CompiledModel {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let a = input(&mut arena, "a", 0, 0, 1);
    let neg = arena.alloc(Node::Scalar(-1.0));
    let neg_a = arena.alloc(Node::Mul(neg, a));
    let rhs = arena.alloc(Node::Mul(neg_a, y));
    let two = arena.alloc(Node::Scalar(2.0));
    let output = arena.alloc(Node::Mul(two, y));
    let threshold = arena.alloc(Node::Scalar(0.4));
    let event = arena.alloc(Node::Sub(y, threshold));
    let mass = identity_mass_matrix(1);
    let mut model = compile_with_sens(&arena, rhs, mass, 1, 1, &[ParamIdx::new(0)]);
    model
        .set_observables(ObservableKind::Outputs, &mut arena, &[output])
        .expect("a valid output");
    model
        .set_observables(ObservableKind::Events, &mut arena, &[event])
        .expect("a valid event");
    model
}
