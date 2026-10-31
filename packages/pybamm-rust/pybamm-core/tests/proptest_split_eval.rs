mod common;

use common::cases::{TangentCase, arb_split_eval_case, targeted_split_eval_cases};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::{
    Arena, CompiledExpr, CsrData, DagRef, Node, NodeId, Shape, SplitCompiledExpr, TangentInputs,
    TypedIr, cse, dce, simplify, tangent_wrt_states, zero_propagate,
};

/// Every seed's sweep on one cached primal matches a monolithic tangent tape.
fn check_split_eval_matches_monolithic(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);

    let mono = CompiledExpr::new(DagRef::new(&arena, tangent_root));
    let mut s_mono = vec![0.0; mono.scratch_len()];
    let split = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    let mut s_split = vec![0.0; split.scratch_len()];
    let mut cache = split.eval_primal(&mut s_split, 0.0, &case.y, &[], &[]);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };
        let mono_result =
            mono.eval_with_tangent(&mut s_mono, 0.0, &case.y, &[], &[], &tangent_inputs);
        assert_bitwise_eq(mono_result, cache.eval_tangent(&tangent_inputs));
    }
}

fn check_repeated_tangent_stable(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);

    let compiled = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut cache = compiled.eval_primal(&mut s, 0.0, &case.y, &[], &[]);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        let first = cache.eval_tangent(&tangent_inputs).to_vec();
        let second = cache.eval_tangent(&tangent_inputs).to_vec();
        assert_bitwise_eq(&first, &second);
    }
}

fn check_split_eval_after_pipeline(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let root = tangent_wrt_states(&mut arena, case.root);
    let root = simplify(&mut arena, root);
    let folded = zero_propagate(DagRef::new(&arena, root));
    let shared = cse(folded.as_ref());
    let pruned = dce(shared.as_ref());

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        let ir_mono = TypedIr::from_dag(pruned.as_ref());
        let mono = CompiledExpr::from_ir(ir_mono);
        let mut s_mono = vec![0.0; mono.scratch_len()];
        let mono_result = mono
            .eval_with_tangent(&mut s_mono, 0.0, &case.y, &[], &[], &tangent_inputs)
            .to_vec();

        let split = SplitCompiledExpr::compile(pruned.as_ref());
        let mut s_split = vec![0.0; split.scratch_len()];
        let mut cache = split.eval_primal(&mut s_split, 0.0, &case.y, &[], &[]);
        let split_result = cache.eval_tangent(&tangent_inputs).to_vec();

        assert_bitwise_eq(&mono_result, &split_result);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn split_eval_matches_monolithic(case in arb_split_eval_case()) {
        check_split_eval_matches_monolithic(&case);
    }

    #[test]
    fn split_eval_repeated_tangent_is_stable(case in arb_split_eval_case()) {
        check_repeated_tangent_stable(&case);
    }

    #[test]
    fn split_eval_after_production_pipeline(case in arb_split_eval_case()) {
        check_split_eval_after_pipeline(&case);
    }
}

/// Unit seeds over `n` states, then `extra` mixed ones.
fn unit_seeds(n: usize, extra: &[&[f64]]) -> Vec<Vec<f64>> {
    let units = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect());
    units
        .chain(extra.iter().map(|seed| seed.to_vec()))
        .collect()
}

/// A `TangentCase` for `build`'s graph over `y.len()` states.
fn case(
    y: Vec<f64>,
    seeds: Vec<Vec<f64>>,
    build: impl FnOnce(&mut Arena) -> NodeId,
) -> TangentCase {
    let mut arena = Arena::new();
    let root = build(&mut arena);
    TangentCase {
        arena,
        root,
        n_states: y.len(),
        y,
        seeds,
        inputs: Vec::new(),
    }
}

/// Hand-built graphs: products, transcendentals, a stencil and a sparse product.
fn hand_built_split_eval_cases() -> Vec<TangentCase> {
    let y_squared = case(
        vec![2.0, 3.0, 4.0],
        unit_seeds(3, &[&[1.0, 1.0, 1.0], &[0.3, -0.7, 1.2]]),
        |arena| {
            let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
            arena.alloc(Node::Mul(y, y))
        },
    );
    let exp_sin = case(
        vec![0.1, 0.5, 1.0],
        unit_seeds(3, &[&[0.5, -0.3, 0.8]]),
        |arena| {
            let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
            let exp_y = arena.alloc(Node::Exp(y));
            let sin_y = arena.alloc(Node::Sin(y));
            arena.alloc(Node::Mul(exp_y, sin_y))
        },
    );
    let n = 10;
    let tridiagonal = case(
        (0..n).map(|i| (i as f64 * 0.3).sin()).collect(),
        unit_seeds(n, &[]),
        |arena| {
            let states: Vec<_> = (0..n)
                .map(|i| {
                    arena.alloc(Node::StateVector {
                        start: i,
                        end: i + 1,
                    })
                })
                .collect();
            let two = arena.alloc(Node::Scalar(2.0));
            let rows = (0..n)
                .map(|i| {
                    let two_mid = arena.alloc(Node::Mul(two, states[i]));
                    let left = states[i.saturating_sub(1)];
                    let right = states[(i + 1).min(n - 1)];
                    let sum_lr = arena.alloc(Node::Add(left, right));
                    arena.alloc(Node::Add(two_mid, sum_lr))
                })
                .collect();
            arena.alloc(Node::Concat(rows))
        },
    );
    let sparse_matmul = case(
        vec![0.1, 0.2, 0.3, 0.4],
        unit_seeds(4, &[&[1.0, 0.0, 1.0, 0.0], &[0.2, -0.4, 0.6, -0.8]]),
        |arena| {
            let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
            let mat = arena.alloc(Node::SparseMatrix(Box::new(
                CsrData::try_new(
                    vec![0, 2, 5, 8, 10],
                    vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
                    vec![1.0; 10],
                    Shape::matrix(4, 4),
                )
                .expect("valid test matrix"),
            )));
            arena.alloc(Node::MatMul(mat, y))
        },
    );
    // Many seeds against one cached primal.
    let scaled_exp = case(
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        (0..20_u32)
            .map(|s| (0..5_u32).map(|i| (f64::from(s + i) * 0.7).sin()).collect())
            .collect(),
        |arena| {
            let y = arena.alloc(Node::StateVector { start: 0, end: 5 });
            let exp_y = arena.alloc(Node::Exp(y));
            let two = arena.alloc(Node::Scalar(2.0));
            arena.alloc(Node::Mul(two, exp_y))
        },
    );
    vec![y_squared, exp_sin, tridiagonal, sparse_matmul, scaled_exp]
}

#[test]
fn targeted_split_eval_cases_pass() {
    for case in targeted_split_eval_cases()
        .into_iter()
        .chain(hand_built_split_eval_cases())
    {
        check_split_eval_matches_monolithic(&case);
        check_repeated_tangent_stable(&case);
        check_split_eval_after_pipeline(&case);
    }
}
