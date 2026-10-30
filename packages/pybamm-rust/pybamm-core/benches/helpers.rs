//! Shared benchmark scaffolding; not every bench target uses every helper.
#![allow(dead_code)]

use pybamm_core::node::{CsrData, Shape};
use pybamm_core::{Arena, DagRef, Node, NodeId, TypedIr, infer_sizes};

pub fn build_coupled(arena: &mut Arena, n_states: usize) -> NodeId {
    let mut terms = Vec::new();
    for i in 0..n_states {
        let y_i = arena.alloc(Node::StateVector {
            start: i,
            end: i + 1,
        });
        let y_next = arena.alloc(Node::StateVector {
            start: (i + 1) % n_states,
            end: (i + 1) % n_states + 1,
        });
        let y_prev = arena.alloc(Node::StateVector {
            start: (i + n_states - 1) % n_states,
            end: (i + n_states - 1) % n_states + 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let y_sq = arena.alloc(Node::Pow(y_i, two));
        let sin_next = arena.alloc(Node::Sin(y_next));
        let prod = arena.alloc(Node::Mul(y_i, y_prev));
        let sum1 = arena.alloc(Node::Add(y_sq, sin_next));
        let term = arena.alloc(Node::Add(sum1, prod));
        terms.push(term);
    }
    arena.alloc(Node::Concat(terms))
}

pub fn identity_mass_matrix(n: usize) -> CsrData {
    CsrData::try_new(
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0; n],
        Shape::matrix(n, n),
    )
    .expect("identity mass matrix is valid")
}

/// Slot reuse of lowering `root`: `(naive, buffer, buffer / naive)`, where `naive` is the
/// element count of a slot-per-node layout.
pub fn slot_stats(arena: &Arena, root: NodeId) -> (usize, usize, f64) {
    let eval_order = arena.topological_order(root);
    let sizes = infer_sizes(arena, &eval_order);
    let naive: usize = eval_order.iter().map(|n| sizes[n.index()]).sum();
    let buffer = TypedIr::from_dag(DagRef::new(arena, root)).buffer_size();
    let ratio = if naive == 0 {
        1.0
    } else {
        buffer as f64 / naive as f64
    };
    (naive, buffer, ratio)
}

/// The fixture's model, with an identity mass matrix when it carries none.
#[cfg(feature = "serialize")]
pub fn compile_fixture(snap: &pybamm_core::DagSnapshot) -> pybamm_core::CompiledModel {
    let mass = snap
        .mass_matrix
        .clone()
        .unwrap_or_else(|| identity_mass_matrix(snap.n_states));
    pybamm_core::CompiledModel::new(
        snap.dag().arena(),
        snap.dag().root(),
        mass,
        snap.n_states,
        snap.packed_input_width,
        pybamm_core::CompiledModelOptions::default(),
    )
    .expect("a valid fixture")
}

#[cfg(feature = "serialize")]
#[derive(Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

/// The `(y, inputs)` the real-model benches evaluate at.
#[cfg(feature = "serialize")]
pub fn fixture_state(n_states: usize, n_params: usize) -> (Vec<f64>, Vec<f64>) {
    let y = (0..n_states)
        .map(|i| 0.01f64.mul_add(i as f64 / n_states as f64, 0.5))
        .collect();
    (y, vec![0.0; n_params])
}

#[cfg(feature = "serialize")]
pub const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "SPM",
        bytes: include_bytes!("fixtures/spm.bin"),
    },
    Fixture {
        name: "SPMe",
        bytes: include_bytes!("fixtures/spme.bin"),
    },
    Fixture {
        name: "DFN",
        bytes: include_bytes!("fixtures/dfn.bin"),
    },
];
