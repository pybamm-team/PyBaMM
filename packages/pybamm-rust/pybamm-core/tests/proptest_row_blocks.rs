//! Grouping split Jacobian rows onto one shared adjoint tape must not change
//! any row's gradient.
//!
//! Two oracles:
//!
//! 1. A grouped tape against a tape holding that row alone. Seeding element `r`
//!    of a shared root must recover exactly what a tape built for row `r` does,
//!    which is the whole contract `ROWS_PER_TAPE` relies on.
//! 2. Both against central finite differences of the parent expression, so a
//!    mistake shared by the two adjoint paths still fails.

mod common;

use common::cases::{DagCase, arb_eval_case, targeted_eval_cases};
use proptest::prelude::*;
use pybamm_core::{Arena, CompiledExpr, NodeId, adjoint::AdjointTape, extract_scalar_rows};

/// Central-difference step, and the tolerance the FD oracle is checked at.
const FD_STEP: f64 = 1e-6;
const FD_RTOL: f64 = 1e-4;
const FD_ATOL: f64 = 1e-6;

/// Assemble `row` of `tape` into a fresh gradient, from its own forward pass.
fn row_gradient(tape: &AdjointTape, row: usize, y: &[f64]) -> Vec<f64> {
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; tape.n_states().max(1)];
    tape.eval_forward(&mut scratch, 0.5, y, &[], &[]);
    tape.assemble_row(&scratch, &mut bar, &mut grad, row);
    grad
}

/// Central finite difference of element `row` of `root` w.r.t. every state.
///
/// `None` unless the difference is finite and has converged -- it is retaken at
/// half the step and kept only if the two agree. `sin(sinh(2y))` at `y = 8.6`
/// moves 31 radians across the stencil, so its quotient is noise of plausible
/// magnitude, which would otherwise read as a broken adjoint.
fn fd_row(arena: &Arena, root: NodeId, row: usize, y: &[f64]) -> Option<Vec<f64>> {
    let expr = CompiledExpr::new(arena, root);
    let mut scratch = vec![0.0; expr.scratch_len()];
    let mut slope_at = |col: usize, step: f64| {
        let mut plus = y.to_vec();
        let mut minus = y.to_vec();
        plus[col] += step;
        minus[col] -= step;
        let f_plus = expr.eval(&mut scratch, 0.5, &plus, &[], &[])[row];
        let f_minus = expr.eval(&mut scratch, 0.5, &minus, &[], &[])[row];
        (f_plus - f_minus) / (2.0 * step)
    };
    (0..y.len())
        .map(|col| {
            let coarse = slope_at(col, FD_STEP);
            // Halving the step cuts an O(h^2) truncation error fourfold, so the
            // two agree only where the difference means something.
            let fine = slope_at(col, FD_STEP / 2.0);
            let converged = fine.is_finite()
                && coarse.is_finite()
                && (coarse - fine).abs() <= FD_ATOL + FD_RTOL * fine.abs();
            converged.then_some(fine)
        })
        .collect()
}

/// Every row of `case`, grouped as the builder likes, against all three oracles.
///
/// Panics on mismatch, so it serves proptest and the targeted tests alike.
/// Returns how many rows the finite-difference oracle ran on, so a caller that
/// means to exercise it can assert that it did.
fn check_row_blocks(case: &DagCase, check_fd: bool) -> usize {
    let mut fd_rows = 0;
    let width = CompiledExpr::new(&case.arena, case.root).output_len();
    if width < 2 {
        return fd_rows; // sharing a tape is only meaningful with rows to share it
    }
    let rows: Vec<usize> = (0..width).collect();
    let Some(block) = extract_scalar_rows(&case.arena, case.root, &rows) else {
        return fd_rows; // an unindexable node declined the split, which is allowed
    };

    {
        let tape = AdjointTape::new(&block.arena, block.root, case.n_states);
        assert_eq!(
            tape.n_rows(),
            block.rows.len(),
            "the tape must hold one element per row it was built for"
        );

        for (element, &row) in block.rows.iter().enumerate() {
            let grouped = row_gradient(&tape, element, &case.y);

            // Oracle 1: a tape holding this row alone must agree with the group.
            let Some(solo_block) = extract_scalar_rows(&case.arena, case.root, &[row]) else {
                continue;
            };
            let solo_tape = AdjointTape::new(&solo_block.arena, solo_block.root, case.n_states);
            let solo = row_gradient(&solo_tape, 0, &case.y);
            assert_close_rows(&grouped, &solo, row, "grouped vs solo");

            // Oracle 2: an independent check that both are the real derivative.
            if check_fd && let Some(fd) = fd_row(&case.arena, case.root, row, &case.y) {
                assert_close_rows(&grouped, &fd, row, "adjoint vs finite difference");
                fd_rows += 1;
            }
        }
    }
    fd_rows
}

/// Compare two gradient rows at the FD tolerance, naming the row on failure.
///
/// Exact equality is checked first so an expression that legitimately overflows
/// to the same infinity on both sides agrees, where `(inf - inf)` would be NaN.
#[track_caller]
fn assert_close_rows(got: &[f64], want: &[f64], row: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "row {row}: {what} length mismatch");
    for (col, (&g, &w)) in got.iter().zip(want).enumerate() {
        // Bit equality first, so a shared overflow to the same infinity agrees
        // where `(inf - inf)` would be NaN. `float_cmp` wants a margin; exactness
        // is the point here.
        if g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()) {
            continue;
        }
        assert!(
            (g - w).abs() <= FD_ATOL + FD_RTOL * w.abs(),
            "row {row}, col {col}: {what}: {g} vs {w}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The whole invariant over arbitrary DAGs. Finite differences are checked
    /// too, since a shared error in both adjoint paths would satisfy oracles 1
    /// and 2 alone.
    #[test]
    fn grouped_rows_match_solo_rows_and_finite_differences(case in arb_eval_case()) {
        check_row_blocks(&case, true);
    }
}

/// The shapes the generator reaches only by luck: a wide `Concat`, a shared
/// sub-expression across rows, and a sparse matmul block.
#[test]
fn targeted_cases_hold_the_row_block_invariants() {
    for case in targeted_eval_cases() {
        check_row_blocks(&case, false);
    }
}

/// A dense block over shared upstream: every row reads every lane of the same
/// expression, which is the shape the split exists for.
#[test]
fn a_dense_block_over_shared_upstream_holds() {
    use pybamm_core::{CsrData, Node, Shape};

    let (n, width) = (24, 6);
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let shared = arena.alloc(Node::Tanh(y));
    let indptr: Vec<usize> = (0..=width).map(|row| row * n).collect();
    let indices: Vec<usize> = (0..width).flat_map(|_| 0..n).collect();
    let data: Vec<f64> = (0..width * n)
        .map(|k| (k as f64).mul_add(0.017, 0.3))
        .collect();
    let matrix = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(indptr, indices, data, Shape::matrix(width, n)).expect("valid matrix"),
    )));
    let root = arena.alloc(Node::MatMul(matrix, shared));

    let case = DagCase {
        arena,
        root,
        y: (0..n)
            .map(|i| (i as f64).mul_add(0.13, 0.2).sin())
            .collect(),
        n_states: n,
    };
    // Well-conditioned by construction, so this is what stops `fd_row`'s
    // convergence gate quietly disabling oracle 2 everywhere.
    assert_eq!(
        check_row_blocks(&case, true),
        width,
        "the finite-difference oracle must run on every row of a well-conditioned block"
    );
}
