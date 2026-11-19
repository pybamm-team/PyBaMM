mod common;

use common::cases::{
    DagCase, Extra, arb_eval_case_with, duplicate_subexpr_case, eval_dag, targeted_eval_cases,
};
use common::numeric_eq::{assert_bitwise_eq, assert_eq_up_to_zero_sign};
use proptest::prelude::*;
use pybamm_core::{Arena, ArrayData, DagRef, Node, NodeId, Shape, cse, simplify, zero_propagate};

/// Zero literals and `ZeroVector`s, which the passes fold, and literal powers, which
/// `simplify` lowers.
const ZEROS_AND_POWERS: &[Extra] = &[Extra::Zeros, Extra::Pow];

fn eval(case: &DagCase) -> Vec<f64> {
    eval_dag(&case.arena, case.root, 0.0, &case.y, &case.inputs)
}

/// Whether every node `case.root` reaches evaluates finite at the case's point.
fn every_intermediate_finite(case: &DagCase) -> bool {
    case.arena
        .topological_order(case.root)
        .into_iter()
        .all(|id| {
            eval_dag(&case.arena, id, 0.0, &case.y, &case.inputs)
                .iter()
                .all(|v| v.is_finite())
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250))]

    /// CSE merges only bitwise-identical nodes, so it is exact, signed zeros and all.
    #[test]
    fn cse_is_bitwise_exact(case in arb_eval_case_with(ZEROS_AND_POWERS)) {
        let shared = cse(DagRef::new(&case.arena, case.root));
        let after = eval_dag(shared.arena(), shared.root(), 0.0, &case.y, &case.inputs);
        assert_bitwise_eq(&eval(&case), &after);
    }

    /// Its documented contract: exact up to the sign of zero where every intermediate is finite.
    #[test]
    fn zero_propagate_is_exact_up_to_the_sign_of_zero(
        case in arb_eval_case_with(ZEROS_AND_POWERS),
    ) {
        prop_assume!(every_intermediate_finite(&case));
        let folded = zero_propagate(DagRef::new(&case.arena, case.root));
        let after = eval_dag(folded.arena(), folded.root(), 0.0, &case.y, &case.inputs);
        assert_eq_up_to_zero_sign(&eval(&case), &after);
    }

    /// Exact up to the sign of zero where every intermediate is finite, literal powers
    /// included: `simplify` lowers one only where the rewrite takes `pow`'s value bitwise.
    #[test]
    fn simplify_is_exact_up_to_the_sign_of_zero(case in arb_eval_case_with(ZEROS_AND_POWERS)) {
        prop_assume!(every_intermediate_finite(&case));
        let mut arena = case.arena.clone();
        let root = simplify(&mut arena, case.root);
        let after = eval_dag(&arena, root, 0.0, &case.y, &case.inputs);
        assert_eq_up_to_zero_sign(&eval(&case), &after);
    }
}

/// Graphs whose rewrites once went wrong, each with the points to check them at.
fn targeted_rewrite_cases() -> Vec<(Arena, NodeId, Vec<Vec<f64>>)> {
    let mut cases = Vec::new();

    // Arrays differing only in the signs of their zeros, which CSE once merged.
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let one = arena.alloc(Node::Scalar(1.0));
    let quotients = [[0.0, -0.0, 3.0], [-0.0, 0.0, 3.0]].map(|data| {
        let data = ArrayData::try_new(data.to_vec(), Shape::vector(3)).expect("valid array");
        let array = arena.alloc(Node::Array(Box::new(data)));
        let product = arena.alloc(Node::Mul(array, y));
        arena.alloc(Node::Div(one, product))
    });
    let root = arena.alloc(Node::Concat(quotients.to_vec()));
    cases.push((arena, root, vec![vec![1.0, 1.0, 1.0], vec![-2.0, 0.5, 4.0]]));

    // A single-branch conditional nested in a privatised cone, which lowering once panicked on.
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let selector = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let always = arena.alloc(Node::Scalar(1.0));
    let sin = arena.alloc(Node::Sin(y));
    let inner = arena.alloc(Node::Conditional {
        selector: always,
        branches: vec![sin],
    });
    let branches = vec![
        arena.alloc(Node::Neg(y)),
        arena.alloc(Node::Exp(inner)),
        arena.alloc(Node::Cos(inner)),
    ];
    let root = arena.alloc(Node::Conditional { selector, branches });
    let points = [1.0, 2.0, 3.0, 0.5, f64::NAN].map(|selector| vec![0.7, selector]);
    cases.push((arena, root, points.to_vec()));

    // A conditional of zeros is zero for every selector, NaN included.
    let mut arena = Arena::new();
    let selector = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let zeros = arena.alloc(Node::ZeroVector { len: 2 });
    let cond = arena.alloc(Node::Conditional {
        selector,
        branches: vec![zeros, zeros],
    });
    let y = arena.alloc(Node::StateVector { start: 1, end: 3 });
    let root = arena.alloc(Node::Add(cond, y));
    let points = [1.0, 2.0, 0.0, f64::NAN].map(|selector| vec![selector, -1.5, 0.25]);
    cases.push((arena, root, points.to_vec()));

    cases
}

#[test]
fn rewrite_passes_preserve_eval_on_targeted_cases() {
    let generated = targeted_eval_cases()
        .into_iter()
        .map(|case| (case.arena, case.root, vec![case.y]));
    for (arena, root, points) in generated.chain(targeted_rewrite_cases()) {
        let shared = cse(DagRef::new(&arena, root));
        let folded = zero_propagate(DagRef::new(&arena, root));
        let mut simplified = arena.clone();
        let simplified_root = simplify(&mut simplified, root);
        for y in &points {
            let original = eval_dag(&arena, root, 0.0, y, &[]);
            let after = [
                eval_dag(shared.arena(), shared.root(), 0.0, y, &[]),
                eval_dag(folded.arena(), folded.root(), 0.0, y, &[]),
                eval_dag(&simplified, simplified_root, 0.0, y, &[]),
            ];
            for after in after {
                assert_bitwise_eq(&original, &after);
            }
        }
    }
}

#[test]
fn cse_reduces_known_duplicate_graph() {
    let case = duplicate_subexpr_case();
    let shared = cse(DagRef::new(&case.arena, case.root));
    assert!(
        shared.arena().len() < case.arena.len(),
        "CSE failed to reduce known duplicates: {} -> {}",
        case.arena.len(),
        shared.arena().len(),
    );
}

/// Exact up to the sign of zero where every intermediate is finite; at a `0 / 0` the fold
/// gives the zero `CasADi` would, where evaluation gives NaN.
#[test]
fn zero_propagate_is_exact_where_every_intermediate_is_finite() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let selector = arena.alloc(Node::StateVector { start: 2, end: 3 });
    let zero = arena.alloc(Node::ZeroVector { len: 2 });
    let zero_matrix = ArrayData::try_new(vec![0.0; 4], Shape::matrix(2, 2)).expect("valid array");
    let zero_matrix = arena.alloc(Node::Array(Box::new(zero_matrix)));
    let quotient = arena.alloc(Node::Div(zero, y));
    let product = arena.alloc(Node::MatMul(zero_matrix, y));
    let branch = arena.alloc(Node::Conditional {
        selector,
        branches: vec![zero, quotient],
    });
    let terms = [quotient, product, branch].map(|term| arena.alloc(Node::Add(term, y)));
    let root = arena.alloc(Node::Concat(terms.to_vec()));

    let folded = zero_propagate(DagRef::new(&arena, root));
    assert!(
        folded.arena().len() < arena.len(),
        "every zero term must fold away"
    );
    for state in [[0.5, -2.0, 1.0], [3.0, 0.25, 2.0], [-1.0, 1.0, 1.5]] {
        let original = eval_dag(&arena, root, 0.0, &state, &[]);
        let after = eval_dag(folded.arena(), folded.root(), 0.0, &state, &[]);
        assert_eq_up_to_zero_sign(&original, &after);
    }

    let singular = [0.0, 1.0, 2.0];
    let original = eval_dag(&arena, root, 0.0, &singular, &[]);
    let after = eval_dag(folded.arena(), folded.root(), 0.0, &singular, &[]);
    assert!(original[0].is_nan(), "0 / 0 + 0 evaluates to NaN");
    assert_eq!(
        after[0].to_bits(),
        0.0_f64.to_bits(),
        "the fold leaves y = 0"
    );
}
