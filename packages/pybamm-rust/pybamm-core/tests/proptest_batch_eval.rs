//! Differential test for the lane-batched observe evaluator.
//!
//! `eval_batch` reorders the interpreter's loops (K lanes per tape pass) but
//! performs the identical per-element float operations in the identical order,
//! so its output must be **bitwise identical** to `k` independent `eval` calls.
//! This is the primary correctness gate: random DAGs from the shared generator
//! plus the targeted cases that cover instructions the random generator does
//! not emit (conditional, interpolant, sparse matmul).

mod common;

use common::cases::{DagCase, arb_eval_case, targeted_eval_cases};
use proptest::prelude::*;
use pybamm_core::{Arena, CompiledExpr, NodeId};

/// Assert `eval_batch` over `k` lanes equals `k` scalar `eval`s, bit for bit.
#[track_caller]
fn assert_batch_matches_scalar(
    arena: &Arena,
    root: NodeId,
    n_states: usize,
    k: usize,
    ts: &[f64],
    y_cols: &[f64],
) {
    let expr = CompiledExpr::new(arena, root);
    let out_len = expr.output_len();

    // Scalar reference: one eval per lane, column-major into `scalar`.
    let mut scalar = vec![0.0_f64; out_len * k];
    let mut s = vec![0.0_f64; expr.scratch_len()];
    for l in 0..k {
        let y = &y_cols[l * n_states..(l + 1) * n_states];
        let res = expr.eval(&mut s, ts[l], y, &[], &[]);
        scalar[l * out_len..(l + 1) * out_len].copy_from_slice(res);
    }

    // Batched: one pass over all lanes.
    let mut batch_scratch = vec![0.0_f64; expr.scratch_len() * k];
    let root_slice = expr
        .eval_batch(&mut batch_scratch, k, ts, y_cols, &[])
        .expect("primal tape must batch-evaluate");

    for l in 0..k {
        for e in 0..out_len {
            let got = root_slice[e * k + l];
            let want = scalar[l * out_len + e];
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "lane {l}, elem {e}: batch {got} != scalar {want} (k={k})"
            );
        }
    }
}

/// `(case, k, ts[k], y_cols[n_states*k])` with `k` sampled from {1, 2, 7, 32}.
fn arb_batch_case() -> impl Strategy<Value = (DagCase, usize, Vec<f64>, Vec<f64>)> {
    arb_eval_case()
        .prop_flat_map(|case| (Just(case), prop::sample::select(vec![1_usize, 2, 7, 32])))
        .prop_flat_map(|(case, k)| {
            let n_states = case.n_states;
            (
                Just(case),
                Just(k),
                prop::collection::vec(-2.0_f64..5.0, k),
                prop::collection::vec(0.1_f64..10.0, n_states * k),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn batch_matches_scalar_random_dag((case, k, ts, y_cols) in arb_batch_case()) {
        assert_batch_matches_scalar(&case.arena, case.root, case.n_states, k, &ts, &y_cols);
    }
}

/// Targeted DAGs cover instructions the random generator never emits
/// (conditional, interpolant, sparse matmul), swept across lane counts
/// including the ragged tail (`k = 1`).
#[test]
fn targeted_cases_batch_match_scalar() {
    for case in targeted_eval_cases() {
        let n = case.n_states;
        for &k in &[1_usize, 2, 7, 32] {
            // Deterministic per-lane state, varied by lane but kept near the
            // case's own domain so guarded ops (sqrt/div) stay in range.
            let mut y_cols = vec![0.0_f64; n * k];
            for l in 0..k {
                for i in 0..n {
                    y_cols[l * n + i] =
                        case.y[i].mul_add(0.03_f64.mul_add(l as f64, 1.0), 0.01 * i as f64);
                }
            }
            let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.1).collect();
            assert_batch_matches_scalar(&case.arena, case.root, n, k, &ts, &y_cols);
        }
    }
}
