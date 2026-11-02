//! Grouping split Jacobian rows onto one shared adjoint tape must not change
//! any row's gradient, checked against solo-row tapes and finite differences.

mod common;

use common::cases::{ALL_EXTRAS, DagCase, arb_eval_case, arb_eval_case_with, targeted_eval_cases};
use proptest::prelude::*;
use pybamm_core::{
    Arena, CompiledExpr, DagRef, Node, NodeId, adjoint::AdjointTape, extract_scalar_rows,
};

const FD_STEP: f64 = 1e-6;
const FD_RTOL: f64 = 1e-4;
const FD_ATOL: f64 = 1e-6;
/// Evaluated round-off is a few ulp rather than one.
const FD_ULP_MARGIN: f64 = 4.0;

fn row_gradient(tape: &AdjointTape, row: usize, y: &[f64], inputs: &[f64]) -> Vec<f64> {
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; y.len().max(1)];
    tape.eval_forward(&mut scratch, 0.5, y, &[], inputs);
    tape.assemble_row(&scratch, &mut bar, &mut grad, row);
    grad
}

/// Central finite difference of element `row` of `root` w.r.t. every state.
///
/// `None` unless every column's quotient is trustworthy at the test tolerance.
fn fd_row(arena: &Arena, root: NodeId, row: usize, y: &[f64], inputs: &[f64]) -> Option<Vec<f64>> {
    let expr = CompiledExpr::new(DagRef::new(arena, root));
    let mut scratch = vec![0.0; expr.scratch_len()];
    let mut slope_at = |col: usize, step: f64| {
        let mut plus = y.to_vec();
        let mut minus = y.to_vec();
        plus[col] += step;
        minus[col] -= step;
        let f_plus = expr.eval(&mut scratch, 0.5, &plus, &[], inputs)[row];
        let f_minus = expr.eval(&mut scratch, 0.5, &minus, &[], inputs)[row];
        (
            (f_plus - f_minus) / (2.0 * step),
            f_plus.abs().max(f_minus.abs()),
        )
    };
    (0..y.len())
        .map(|col| {
            let (coarse, _) = slope_at(col, FD_STEP);
            // Coarse and fine agree only where truncation error is negligible.
            let (fine, scale) = slope_at(col, FD_STEP / 2.0);
            let judged_at = FD_ATOL + FD_RTOL * fine.abs();
            // Smallest slope the stencil can resolve out of values this size.
            let cancellation = FD_ULP_MARGIN * f64::EPSILON * scale / FD_STEP;
            let resolved = fine.is_finite()
                && coarse.is_finite()
                && (coarse - fine).abs() <= judged_at
                && cancellation <= judged_at;
            resolved.then_some(fine)
        })
        .collect()
}

/// Checks every row of `case`; returns how many rows the FD oracle ran on.
fn check_row_blocks(case: &DagCase, check_fd: bool) -> usize {
    let mut fd_rows = 0;
    let width = CompiledExpr::new(DagRef::new(&case.arena, case.root)).output_len();
    if width < 2 {
        return fd_rows;
    }
    let rows: Vec<usize> = (0..width).collect();
    let block = extract_scalar_rows(DagRef::new(&case.arena, case.root), &rows);

    let tape = AdjointTape::new(block.dag.as_ref(), case.n_states);
    assert_eq!(
        tape.n_rows(),
        block.rows.len(),
        "the tape must hold one element per row it was built for"
    );

    for (element, &row) in block.rows.iter().enumerate() {
        let grouped = row_gradient(&tape, element, &case.y, &case.inputs);

        let solo_block = extract_scalar_rows(DagRef::new(&case.arena, case.root), &[row]);
        let solo_tape = AdjointTape::new(solo_block.dag.as_ref(), case.n_states);
        let solo = row_gradient(&solo_tape, 0, &case.y, &case.inputs);
        assert_close_rows(&grouped, &solo, row, "grouped vs solo");

        let whole = unextracted_row_gradient(case, row);
        assert_close_rows(&grouped, &whole, row, "grouped vs unextracted");

        if check_fd && let Some(fd) = fd_row(&case.arena, case.root, row, &case.y, &case.inputs) {
            // A singular partial times a zero one, as in `sqrt(|y - y|)`, is NaN without
            // extraction too, though the row is flat.
            let fd: Vec<f64> = fd
                .iter()
                .zip(&whole)
                .map(|(&fd, &whole)| if whole.is_nan() { whole } else { fd })
                .collect();
            assert_close_rows(&grouped, &fd, row, "adjoint vs finite difference");
            fd_rows += 1;
        }
    }
    fd_rows
}

/// The reverse gradient of element `row` of `case`'s root, read off the root unextracted.
fn unextracted_row_gradient(case: &DagCase, row: usize) -> Vec<f64> {
    let mut arena = case.arena.clone();
    let element = arena.alloc(Node::Index {
        child: case.root,
        start: row,
        end: row + 1,
    });
    let tape = AdjointTape::new(DagRef::new(&arena, element), case.n_states);
    row_gradient(&tape, 0, &case.y, &case.inputs)
}

/// Compare two gradient rows at the FD tolerance, naming the row on failure.
#[track_caller]
fn assert_close_rows(got: &[f64], want: &[f64], row: usize, what: &str) {
    assert_eq!(got.len(), want.len(), "row {row}: {what} length mismatch");
    for (col, (&g, &w)) in got.iter().zip(want).enumerate() {
        // Bit equality first, so a shared overflow to the same infinity agrees.
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

    #[test]
    fn grouped_rows_match_solo_rows_and_finite_differences(case in arb_eval_case()) {
        check_row_blocks(&case, true);
    }

    /// Interpolants and reductions, which a row slices from one shared copy.
    #[test]
    fn grouped_rows_hold_with_every_extra(case in arb_eval_case_with(ALL_EXTRAS)) {
        check_row_blocks(&case, true);
    }
}

#[test]
fn targeted_cases_hold_the_row_block_invariants() {
    for case in targeted_eval_cases() {
        check_row_blocks(&case, false);
    }
}

/// Regression seed `sinh(erf(y2) * -y)`: `erf(4.76)` is flat to within a rounding step.
#[test]
fn a_stencil_flat_to_within_a_rounding_step_is_declined() {
    let mut arena = Arena::new();
    let scalar = arena.alloc(Node::StateVector { start: 2, end: 3 });
    let erf = arena.alloc(Node::Erf(scalar));
    let states = arena.alloc(Node::StateVector { start: 0, end: 6 });
    let negated = arena.alloc(Node::Neg(states));
    let scaled = arena.alloc(Node::Mul(erf, negated));
    let root = arena.alloc(Node::Sinh(scaled));
    let y = vec![
        0.1,
        0.1,
        4.755_430_192_319_827,
        9.468_766_455_112_123,
        0.1,
        0.1,
    ];

    assert!(
        fd_row(&arena, root, 3, &y, &[]).is_none(),
        "a column whose stencil cannot resolve a slope must not become an oracle"
    );
    let case = DagCase {
        arena,
        root,
        y,
        n_states: 6,
        inputs: Vec::new(),
    };
    assert_eq!(
        check_row_blocks(&case, true),
        5,
        "only the amplified row loses its finite differences; the other five keep them"
    );
}

#[test]
fn a_dense_block_over_shared_upstream_holds() {
    use pybamm_core::{CsrData, Shape};

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
        inputs: Vec::new(),
    };
    // Guards against `fd_row`'s gate quietly disabling the FD oracle everywhere.
    assert_eq!(
        check_row_blocks(&case, true),
        width,
        "the finite-difference oracle must run on every row of a well-conditioned block"
    );
}
