//! Shared benchmark scaffolding; not every bench target uses every helper.
#![allow(dead_code)]

use pybamm_core::node::{CsrData, Shape};
use pybamm_core::{Arena, Node, NodeId};

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

#[cfg(feature = "serialize")]
#[derive(Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

/// The `(y, inputs)` the real-model benches evaluate at. Shared so the eval and
/// assembly numbers stay comparable.
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
