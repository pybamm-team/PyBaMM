//! Bitwise-equivalence tests for split primal/tangent evaluation.
//!
//! Verifies that `eval_primal()` + `PrimalCache::eval_tangent()` produces identical
//! results to `eval_with_tangent()` for various expressions and seed vectors.
use pybamm_core::arena::Arena;
use pybamm_core::eval::{CompiledExpr, TangentInputs};
use pybamm_core::ir::TypedIr;
use pybamm_core::node::{CsrData, Node, Shape};
use pybamm_core::tangent::tangent_wrt_states;

/// Build a tangent expression for `expr` w.r.t. all states, then compile it
/// both with standard IR and split-eval IR, and verify bitwise equality
/// across multiple seed vectors.
fn check_split_eval_equivalence(
    arena: &Arena,
    expr: pybamm_core::arena::NodeId,
    _n_states: usize,
    y: &[f64],
    seeds: &[Vec<f64>],
) {
    let mut diff_arena = arena.clone();
    let jac_y = tangent_wrt_states(&mut diff_arena, expr);

    let standard_ir = TypedIr::from_arena(&diff_arena, jac_y);
    let split_ir = TypedIr::from_arena_split_eval(&diff_arena, jac_y);

    assert!(
        split_ir.split_eval_info().is_some(),
        "split IR should have SplitEvalInfo"
    );

    let standard_expr = CompiledExpr::from_ir(standard_ir);
    let split_expr = CompiledExpr::from_ir(split_ir);
    let mut s_standard = vec![0.0; standard_expr.scratch_len()];
    let mut s_split = vec![0.0; split_expr.scratch_len()];

    // Evaluate primal once
    let mut cache = split_expr.eval_primal(&mut s_split, 0.0, y, &[], &[]);

    for (seed_idx, seed) in seeds.iter().enumerate() {
        let tangent = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        let standard_result =
            standard_expr.eval_with_tangent(&mut s_standard, 0.0, y, &[], &[], &tangent);
        let split_result = cache.eval_tangent(&tangent);

        assert_eq!(
            standard_result.len(),
            split_result.len(),
            "Output length mismatch at seed {seed_idx}"
        );

        for (i, (s, sp)) in standard_result.iter().zip(split_result.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                sp.to_bits(),
                "Bitwise mismatch at output[{i}], seed {seed_idx}: standard={s}, split={sp}"
            );
        }
    }
}

#[test]
fn test_split_eval_y_squared() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let expr = arena.alloc(Node::Mul(y, y));

    let seeds = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![0.3, -0.7, 1.2],
    ];

    let y_vals = [2.0, 3.0, 4.0];
    check_split_eval_equivalence(&arena, expr, 3, &y_vals, &seeds);
}

#[test]
fn test_split_eval_exp_sin() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let exp_y = arena.alloc(Node::Exp(y));
    let sin_y = arena.alloc(Node::Sin(y));
    let expr = arena.alloc(Node::Mul(exp_y, sin_y));

    let seeds = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.5, -0.3, 0.8],
    ];

    let y_vals = [0.1, 0.5, 1.0];
    check_split_eval_equivalence(&arena, expr, 3, &y_vals, &seeds);
}

#[test]
fn test_split_eval_tridiagonal() {
    let n = 10usize;
    let mut arena = Arena::new();

    let svecs: Vec<_> = (0..n)
        .map(|i| {
            arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            })
        })
        .collect();

    let two = arena.alloc(Node::Scalar(2.0));

    let rows: Vec<_> = (0..n)
        .map(|i| {
            let left = svecs[i.saturating_sub(1)];
            let mid = svecs[i];
            let right = svecs[(i + 1).min(n - 1)];

            let two_mid = arena.alloc(Node::Mul(two, mid));
            let sum_lr = arena.alloc(Node::Add(left, right));
            arena.alloc(Node::Add(two_mid, sum_lr))
        })
        .collect();

    let rhs = arena.alloc(Node::Concat(rows));

    let seeds: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut s = vec![0.0; n];
            s[i] = 1.0;
            s
        })
        .collect();

    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    check_split_eval_equivalence(&arena, rhs, n, &y, &seeds);
}

#[test]
fn test_split_eval_sparse_matmul() {
    let mut arena = Arena::new();
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
    let expr = arena.alloc(Node::MatMul(mat, y));

    let seeds = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.2, -0.4, 0.6, -0.8],
    ];

    let y_vals = [0.1, 0.2, 0.3, 0.4];
    check_split_eval_equivalence(&arena, expr, 4, &y_vals, &seeds);
}

#[test]
fn test_split_eval_multi_seed_stability() {
    // Verify calling eval_tangent many times after a single eval_primal
    // produces correct results every time (no buffer corruption).
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 5 });
    let exp_y = arena.alloc(Node::Exp(y));
    let two = arena.alloc(Node::Scalar(2.0));
    let expr = arena.alloc(Node::Mul(two, exp_y));

    let mut diff_arena = arena.clone();
    let jac_y = tangent_wrt_states(&mut diff_arena, expr);

    let standard_ir = TypedIr::from_arena(&diff_arena, jac_y);
    let split_ir = TypedIr::from_arena_split_eval(&diff_arena, jac_y);

    let y_vals = [0.1, 0.2, 0.3, 0.4, 0.5];

    let standard_expr = CompiledExpr::from_ir(standard_ir);
    let split_expr = CompiledExpr::from_ir(split_ir);
    let mut s_standard = vec![0.0; standard_expr.scratch_len()];
    let mut s_split = vec![0.0; split_expr.scratch_len()];

    let mut cache = split_expr.eval_primal(&mut s_split, 0.0, &y_vals, &[], &[]);

    // Run 20 different seeds after a single primal eval
    for seed_idx in 0..20u64 {
        let seed: Vec<f64> = (0_u32..5)
            .map(|i| ((seed_idx as f64 + f64::from(i)) * 0.7).sin())
            .collect();

        let tangent = TangentInputs {
            dy: Some(&seed),
            dp: None,
        };

        let standard_result =
            standard_expr.eval_with_tangent(&mut s_standard, 0.0, &y_vals, &[], &[], &tangent);
        let split_result = cache.eval_tangent(&tangent);

        for (i, (s, sp)) in standard_result.iter().zip(split_result.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                sp.to_bits(),
                "Bitwise mismatch at output[{i}], seed {seed_idx}: standard={s}, split={sp}"
            );
        }
    }
}

#[test]
fn test_split_eval_jacobian_assembly() {
    // Verify the full Jacobian assembly path with split eval works correctly.
    // Numerical correctness is verified by test_coloring_correctness (finite differences).
    use pybamm_core::model::ModelEvaluator;
    use pybamm_core::node::{CsrData, Shape};

    let n = 20usize;
    let mut arena = Arena::new();

    let svecs: Vec<_> = (0..n)
        .map(|i| {
            arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            })
        })
        .collect();

    let two = arena.alloc(Node::Scalar(2.0));

    let rows: Vec<_> = (0..n)
        .map(|i| {
            let left = svecs[i.saturating_sub(1)];
            let mid = svecs[i];
            let right = svecs[(i + 1).min(n - 1)];

            let two_mid = arena.alloc(Node::Mul(two, mid));
            let sum_lr = arena.alloc(Node::Add(left, right));
            arena.alloc(Node::Add(two_mid, sum_lr))
        })
        .collect();

    let rhs = arena.alloc(Node::Concat(rows));

    let mass = CsrData::try_new(
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0; n],
        Shape::matrix(n, n),
    )
    .expect("valid identity mass matrix");

    let mut model = ModelEvaluator::new(&arena, rhs, mass, n, 0);
    model.set_cj(0.0);

    // Just verify assembly completes without panicking.
    // Numerical correctness is tested in test_coloring_correctness.
    let nnz = model.nnz();
    let mut jac = vec![0.0_f64; nnz];
    let y = vec![0.1_f64; n];
    model.assemble_jacobian_csc_into_coloring(0.0, &y, &[], &mut jac);

    // Verify non-trivial result (not all zeros)
    let nonzero_count = jac.iter().filter(|&&v| v.abs() > 1e-15).count();
    assert!(
        nonzero_count > 0,
        "Jacobian should have nonzero entries, got {nonzero_count}"
    );
}
