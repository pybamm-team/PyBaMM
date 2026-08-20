//! Correctness tests: verify coloring-based Jacobian assembly against
//! central finite differences.
//!
//! This module provides an independent ground truth for Jacobian correctness
//! that does not depend on any symbolic AD infrastructure.

use pybamm_core::NodeId;
use pybamm_core::arena::Arena;
use pybamm_core::eval::CompiledExpr;
use pybamm_core::jacobian::{JacobianData, JacobianScratch};
use pybamm_core::model::ModelEvaluator;
use pybamm_core::node::{CsrData, InterpolantData, Node, Shape};

/// Assemble coloring Jacobian and compare each entry against central
/// finite differences.
///
/// Strategy:
/// 1. Assemble coloring Jacobian → dense n×n matrix (scattered from CSC)
/// 2. For each column j, compute FD column via 2 RHS evaluations
/// 3. Compare entry-by-entry
fn compare_coloring_vs_fd(
    model: &mut ModelEvaluator,
    y: &[f64],
    inputs: &[f64],
    h: f64,
    rtol: f64,
) {
    let n = model.n_states();
    let nnz = model.nnz();

    // Assemble coloring Jacobian into CSC buffer
    let mut jac_csc = vec![0.0_f64; nnz];
    model.assemble_jacobian_csc_into_coloring(0.0, y, inputs, &mut jac_csc);

    // Scatter CSC into dense matrix
    let csc_sparsity = model.csc_sparsity();
    let mut dense = vec![vec![0.0_f64; n]; n];
    for (col, colptr) in csc_sparsity.colptr.windows(2).enumerate() {
        let range_start = colptr[0];
        let range_end = colptr[1];
        for (local, &val) in jac_csc[range_start..range_end].iter().enumerate() {
            let row = csc_sparsity.rowind[range_start + local];
            dense[row][col] = val;
        }
    }

    // Compute FD columns and compare
    // NOTE: eval_rhs writes to output buffer, does NOT return a value
    let mut f_plus = vec![0.0_f64; n];
    let mut f_minus = vec![0.0_f64; n];

    for col in 0..n {
        let mut y_plus = y.to_vec();
        let mut y_minus = y.to_vec();
        let scale = h.max(1e-10);
        y_plus[col] += scale;
        y_minus[col] -= scale;

        model.eval_rhs(0.0, &y_plus, inputs, &mut f_plus);
        model.eval_rhs(0.0, &y_minus, inputs, &mut f_minus);

        for row in 0..n {
            let fd_val = (f_plus[row] - f_minus[row]) / (2.0 * scale);
            let coloring_val = dense[row][col];
            let err = (fd_val - coloring_val).abs();
            let scale_val = fd_val.abs().max(coloring_val.abs()).max(1e-15);
            assert!(
                err / scale_val < rtol,
                "Jacobian({row},{col}): coloring={coloring_val}, fd={fd_val}, err={err}, rel={}",
                err / scale_val
            );
        }
    }
}

/// Build a banded (tridiagonal) model: `f_i = x_{i-1} + 2*x_i + x_{i+1}`
fn build_banded_model(n: usize) -> ModelEvaluator {
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
    ModelEvaluator::new(&arena, rhs, mass, n, 0)
}

/// Build a nonlinear model with cross-coupling:
///   `f_i = sin(y_i) + y_{i-1} * y_{i+1}`  (with boundary clamping)
fn build_nonlinear_model(n: usize) -> ModelEvaluator {
    let mut arena = Arena::new();
    let svecs: Vec<_> = (0..n)
        .map(|i| {
            arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            })
        })
        .collect();

    let rows: Vec<_> = (0..n)
        .map(|i| {
            let self_term = arena.alloc(Node::Sin(svecs[i]));
            let left = svecs[i.saturating_sub(1)];
            let right = svecs[(i + 1).min(n - 1)];
            let coupling = arena.alloc(Node::Mul(left, right));
            arena.alloc(Node::Add(self_term, coupling))
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
    ModelEvaluator::new(&arena, rhs, mass, n, 0)
}

/// Build a model with sparse matrix multiplication:
///   `f = A @ y` where `A` is a constant sparse matrix
fn build_sparse_matmul_model(n: usize) -> ModelEvaluator {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });

    // Build a banded sparse matrix A (tridiagonal with 1, 2, 1)
    let mut indptr = vec![0usize; n + 1];
    let mut indices = Vec::with_capacity(n * 3);
    let mut data = Vec::with_capacity(n * 3);

    for (row, ptr) in indptr.iter_mut().enumerate().take(n) {
        *ptr = data.len();
        if row > 0 {
            indices.push(row - 1);
            data.push(1.0);
        }
        indices.push(row);
        data.push(2.0);
        if row + 1 < n {
            indices.push(row + 1);
            data.push(1.0);
        }
    }
    indptr[n] = data.len();

    let sparse = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(indptr, indices, data, Shape::matrix(n, n)).expect("valid test matrix"),
    )));
    let rhs = arena.alloc(Node::MatMul(sparse, y));

    let mass = CsrData::try_new(
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0; n],
        Shape::matrix(n, n),
    )
    .expect("valid identity mass matrix");
    ModelEvaluator::new(&arena, rhs, mass, n, 0)
}

#[test]
fn test_coloring_vs_fd_banded_5() {
    let n = 5;
    let mut model = build_banded_model(n);
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.5, 0.1).sin()).collect();

    compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
}

#[test]
fn test_coloring_vs_fd_banded_50() {
    let n = 50;
    let mut model = build_banded_model(n);
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect();

    compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
}

#[test]
fn test_coloring_vs_fd_nonlinear_10() {
    let n = 10;
    let mut model = build_nonlinear_model(n);
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.3, 0.5).sin()).collect();

    compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
}

#[test]
fn test_coloring_vs_fd_sparse_matmul_8() {
    let n = 8;
    let mut model = build_sparse_matmul_model(n);
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.4, 0.2).sin()).collect();

    compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
}

#[test]
fn test_coloring_vs_fd_multiple_points() {
    let n = 15;
    let mut model = build_nonlinear_model(n);
    model.set_cj(0.0);

    for seed in [42u64, 137, 2718, 31415, 99991] {
        let y: Vec<f64> = (0..n)
            .map(|i| ((seed as f64 + i as f64) * 0.1).sin())
            .collect();

        compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
    }
}

#[test]
fn test_coloring_vs_fd_with_mass_cj() {
    // With cj != 0 the assembled Jacobian is df/dy - cj*M, while FD gives df/dy,
    // so cj*M is subtracted here.
    let n = 10;
    let mut model = build_banded_model(n);
    let cj = 0.5;
    model.set_cj(cj);

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.3, 0.5).sin()).collect();

    let nnz = model.nnz();
    let mut jac_csc = vec![0.0_f64; nnz];
    model.assemble_jacobian_csc_into_coloring(0.0, &y, &[], &mut jac_csc);

    // Compute FD (using output buffer pattern)
    let h = 1e-7;
    let mut dense_fd = vec![vec![0.0_f64; n]; n];
    let mut f_plus = vec![0.0_f64; n];
    let mut f_minus = vec![0.0_f64; n];

    for col in 0..n {
        let mut y_plus = y.clone();
        let mut y_minus = y.clone();
        y_plus[col] += h;
        y_minus[col] -= h;
        model.eval_rhs(0.0, &y_plus, &[], &mut f_plus);
        model.eval_rhs(0.0, &y_minus, &[], &mut f_minus);
        for row in 0..n {
            dense_fd[row][col] = (f_plus[row] - f_minus[row]) / (2.0 * h);
        }
    }

    // Subtract cj * identity from FD (mass matrix is identity)
    for (i, row) in dense_fd.iter_mut().enumerate() {
        row[i] -= cj;
    }

    // Scatter CSC into dense
    let csc_sparsity = model.csc_sparsity();
    let mut dense_csc = vec![vec![0.0_f64; n]; n];
    for (col, colptr) in csc_sparsity.colptr.windows(2).enumerate() {
        let range_start = colptr[0];
        let range_end = colptr[1];
        for (local, &val) in jac_csc[range_start..range_end].iter().enumerate() {
            let row = csc_sparsity.rowind[range_start + local];
            dense_csc[row][col] = val;
        }
    }

    for row in 0..n {
        for col in 0..n {
            let err = (dense_csc[row][col] - dense_fd[row][col]).abs();
            let scale = dense_csc[row][col]
                .abs()
                .max(dense_fd[row][col].abs())
                .max(1e-15);
            assert!(
                err / scale < 1e-5,
                "Jacobian({row},{col}) with cj={cj}: assembled={}, fd={}",
                dense_csc[row][col],
                dense_fd[row][col]
            );
        }
    }
}

#[test]
fn test_coloring_vs_fd_identity_jacobian() {
    // f_i = y_i → Jacobian = I
    let n = 6;
    let mut arena = Arena::new();
    let svecs: Vec<_> = (0..n)
        .map(|i| {
            arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            })
        })
        .collect();
    let rhs = arena.alloc(Node::Concat(svecs));

    let mass = CsrData::try_new(
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0; n],
        Shape::matrix(n, n),
    )
    .expect("valid identity mass matrix");
    let mut model = ModelEvaluator::new(&arena, rhs, mass, n, 0);
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.5, 0.1).sin()).collect();

    compare_coloring_vs_fd(&mut model, &y, &[], 1e-7, 1e-5);
}

// Dense-row split (loop A). These tests drive `JacobianData` directly: build via
// `new_wrt_states` and finite-difference its `assemble_csc_into` output.

/// Dense-row fixture: `sin(y)` elementwise (a diagonal block) concatenated with
/// one scalar row `ones_row(1 x n) @ (y*y)` that depends on every state — the
/// SPMe/vaas shape of a dense algebraic row atop a sparse structure.
fn build_dense_row_expr(n: usize) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let elt = arena.alloc(Node::Sin(y)); // len n, diagonal
    let gy = arena.alloc(Node::Mul(y, y)); // len n, diagonal, nonlinear
    let ones = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .expect("valid test matrix"),
    )));
    let dense = arena.alloc(Node::MatMul(ones, gy)); // len 1, depends on all n
    let root = arena.alloc(Node::Concat(vec![elt, dense]));
    (arena, root, n)
}

/// Same shape but the dense row lives inside a length-2 vector block (a `2 x n`
/// matmul), so no scalar sub-node holds it and the split has to synthesise one.
fn build_dense_row_vector_block_expr(n: usize) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let elt = arena.alloc(Node::Sin(y));
    let gy = arena.alloc(Node::Mul(y, y));
    // 2 x n: row 0 dense (all cols), row 1 sparse (col 0 only).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.push(0);
    let a = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n, n + 1],
            indices,
            vec![1.0; n + 1],
            Shape::matrix(2, n),
        )
        .expect("valid test matrix"),
    )));
    let block = arena.alloc(Node::MatMul(a, gy)); // len 2, row 0 dense
    let root = arena.alloc(Node::Concat(vec![elt, block]));
    (arena, root, n)
}

/// Assemble `jac`'s CSC values at `t = 0.5` through the production driver, at the
/// lane width this artifact's tape would actually run.
fn assemble_csc(jac: &JacobianData, y: &[f64]) -> Vec<f64> {
    let layout = jac.layout();
    let mut scratch = JacobianScratch::new(jac);
    let mut data = vec![0.0; layout.n_slots()];
    jac.assemble_into(&mut scratch, layout, 0.5, y, &[], &[], &mut data);
    data
}

/// Assemble the `JacobianData` via loop A and compare every CSC entry against
/// central finite differences of the primal expression.
fn fd_check_jacobian_data(arena: &Arena, root: NodeId, jac: &JacobianData, y: &[f64]) {
    let primal = CompiledExpr::new(arena, root);
    let data = assemble_csc(jac, y);

    let mut s = vec![0.0; primal.scratch_len()];
    let eps = 1e-6;
    for col in 0..jac.n_cols() {
        let mut yp = y.to_vec();
        yp[col] += eps;
        let mut ym = y.to_vec();
        ym[col] -= eps;
        let fp = primal.eval(&mut s, 0.5, &yp, &[], &[]).to_vec();
        let fm = primal.eval(&mut s, 0.5, &ym, &[], &[]).to_vec();
        let (lo, hi) = (jac.csc().colptr[col], jac.csc().colptr[col + 1]);
        for (&dk, &row) in data[lo..hi].iter().zip(&jac.csc().rowind[lo..hi]) {
            let fd = (fp[row] - fm[row]) / (2.0 * eps);
            assert!(
                (dk - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "entry ({row},{col}): assembled {dk} vs fd {fd}"
            );
        }
    }
}

/// A block of `width` fully dense rows over `n` states, plus a diagonal
/// remainder: the shape a 2-D current collector produces, where one dense row
/// per collector node sits inside a single vector-valued block.
///
/// `coupled` builds the block's interior, which is what decides whether the
/// rows can be extracted at all.
fn build_dense_block_expr(
    n: usize,
    width: usize,
    coupled: impl FnOnce(&mut Arena, NodeId) -> NodeId,
) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let diagonal = arena.alloc(Node::Sin(y));
    let coupled = coupled(&mut arena, y);
    // width x n, every entry stored: each block row reaches every state.
    let indptr: Vec<usize> = (0..=width).map(|row| row * n).collect();
    let indices: Vec<usize> = (0..width).flat_map(|_| 0..n).collect();
    let data: Vec<f64> = (0..width * n)
        .map(|k| (k as f64).mul_add(0.03, 0.5))
        .collect();
    let matrix = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(indptr, indices, data, Shape::matrix(width, n))
            .expect("valid test matrix"),
    )));
    let block = arena.alloc(Node::MatMul(matrix, coupled));
    let root = arena.alloc(Node::Concat(vec![diagonal, block]));
    (arena, root, n)
}

/// Assemble `jac` into a dense matrix, for comparing two builds of the same
/// derivative entry by entry.
fn assemble_dense(jac: &JacobianData, y: &[f64]) -> Vec<Vec<f64>> {
    let data = assemble_csc(jac, y);
    let mut dense = vec![vec![0.0; jac.n_cols()]; jac.n_rows()];
    for (col, span) in jac.csc().colptr.windows(2).enumerate() {
        for k in span[0]..span[1] {
            dense[jac.csc().rowind[k]][col] = data[k];
        }
    }
    dense
}

#[test]
fn test_dense_block_splits_every_row_and_matches_the_unsplit_build() {
    // A minority of the states, as a 2-D collector's block is: the regime where
    // deleting one colour per state pays for the reverse passes.
    let (n, width) = (64, 16);
    let (arena, root, n_states) =
        build_dense_block_expr(n, width, |arena, y| arena.alloc(Node::Mul(y, y)));
    let n_rows = CompiledExpr::new(&arena, root).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    assert_eq!(jac.n_dense_rows(), width, "the whole plateau must split");
    assert_eq!(jac.n_candidate_rows(), width);
    assert!(
        jac.coloring().n_colors <= width + 1,
        "the split must leave the sparse remainder cheap to colour, got {}",
        jac.coloring().n_colors
    );

    // A full-column subset builds the same derivative with splitting off, so
    // it is the reference the split must reproduce.
    let all: Vec<usize> = (0..n_states).collect();
    let reference = JacobianData::new_wrt_state_subset(&arena, root, n_rows, n_states, &all);
    assert!(reference.dense_rows().is_empty());
    assert!(reference.coloring().n_colors > jac.coloring().n_colors);

    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64).mul_add(0.13, 0.7).sin())
        .collect();
    let split = assemble_dense(&jac, &y);
    let unsplit = assemble_dense(&reference, &y);
    for (row, (split_row, unsplit_row)) in split.iter().zip(&unsplit).enumerate() {
        for (col, (&a, &b)) in split_row.iter().zip(unsplit_row).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().mul_add(1.0, 1.0),
                "entry ({row},{col}): split {a} vs unsplit {b}"
            );
        }
    }
    fd_check_jacobian_data(&arena, root, &jac, &y);
}

/// An interpolant has no cheap indexed form, so a block containing one keeps
/// the wide colouring. The decline has to stay visible in the telemetry.
#[test]
fn test_unextractable_dense_block_declines_visibly() {
    let (n, width) = (64, 16);
    let (arena, root, n_states) = build_dense_block_expr(n, width, |arena, y| {
        arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                InterpolantData::try_new(vec![-2.0, 0.0, 2.0], vec![-1.0, 0.5, 3.0])
                    .expect("valid table"),
            ),
            child: y,
        })
    });

    let n_rows = CompiledExpr::new(&arena, root).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);
    assert!(
        jac.dense_rows().is_empty(),
        "an interpolant must stop the walk"
    );
    assert_eq!(
        jac.n_candidate_rows(),
        width,
        "a declined split must still report what it wanted"
    );

    let y_values: Vec<f64> = (0..n)
        .map(|i| (i as f64).mul_add(0.11, 0.2).sin())
        .collect();
    fd_check_jacobian_data(&arena, root, &jac, &y_values);
}

#[test]
fn test_dense_row_split_matches_fd() {
    let n = 30;
    let (arena, root, n_states) = build_dense_row_expr(n);
    let n_rows = CompiledExpr::new(&arena, root).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    // 30-nnz row >= DENSE_ROW_MIN_NNZ, extracts to a scalar block, and the
    // reduced coloring (1) strictly beats the full coloring (30) -> split.
    assert_eq!(jac.n_dense_rows(), 1, "dense row must be split out");
    assert!(
        jac.coloring().n_colors <= 4,
        "residual coloring must reflect the sparse structure, got {}",
        jac.coloring().n_colors
    );

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 0.3).sin()).collect();
    fd_check_jacobian_data(&arena, root, &jac, &y);
}

#[test]
fn test_dense_conditional_ignores_invalid_inactive_branch() {
    let mut arena = Arena::new();
    let n = 20;
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let ones = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .unwrap(),
    )));
    let active = arena.alloc(Node::MatMul(ones, y));
    let y0 = arena.alloc(Node::Index {
        child: y,
        start: 0,
        end: 1,
    });
    let inactive = arena.alloc(Node::Sqrt(y0));
    let selector = arena.alloc(Node::Scalar(1.0));
    let root = arena.alloc(Node::Conditional {
        selector,
        branches: vec![active, inactive],
    });
    let jac = JacobianData::new_wrt_states(&arena, root, 1, n);
    assert_eq!(jac.n_dense_rows(), 1, "dense row must use reverse AD");

    fd_check_jacobian_data(&arena, root, &jac, &vec![-1.0; n]);
}

// Dense-row split, model-level loops C/D. A green `compare_coloring_vs_fd` proves
// the model consumes `dense_rows` rather than aliasing the reduced coloring.

/// Square DAE: `f_i = sin(y_i)` for the first `n-1` (differential) rows and
/// `f_{n-1} = sum_j y_j^2` (algebraic dense row over all `n` states).
fn build_dense_row_model(n: usize) -> ModelEvaluator {
    let mut arena = Arena::new();
    let y_full = arena.alloc(Node::StateVector { start: 0, end: n });
    let y_head = arena.alloc(Node::StateVector {
        start: 0,
        end: n - 1,
    });
    let diff = arena.alloc(Node::Sin(y_head));
    let gy = arena.alloc(Node::Mul(y_full, y_full));
    let ones = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .expect("valid test matrix"),
    )));
    let dense = arena.alloc(Node::MatMul(ones, gy));
    let rhs = arena.alloc(Node::Concat(vec![diff, dense]));
    let mass = CsrData::try_new(
        (0..n).chain(std::iter::once(n - 1)).collect(),
        (0..n - 1).collect(),
        vec![1.0; n - 1],
        Shape::matrix(n, n),
    )
    .expect("valid test mass matrix");
    ModelEvaluator::new(&arena, rhs, mass, n, 0)
}

#[test]
fn test_coloring_vs_fd_dense_row_20() {
    let n = 20;
    let mut model = build_dense_row_model(n);
    // The split must be active: reduced coloring is far below the dense row's nnz.
    assert!(
        model.coloring().n_colors <= 4,
        "dense-row split must reduce the coloring, got {}",
        model.coloring().n_colors
    );
    model.set_cj(0.0);

    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64).mul_add(0.1, 0.3).sin() + 0.7)
        .collect();
    compare_coloring_vs_fd(&mut model, &y, &[], 1e-6, 1e-5);
}

#[test]
fn test_dense_row_stats_report_the_split_and_its_tape() {
    let model = build_dense_row_model(20);
    let stats = model.jacobian_stats();
    assert_eq!(stats.n_dense_rows, 1);
    // The split's compiled-memory cost, which nothing else asserts.
    assert!(stats.dense_row_tape_instructions > 0);
}

#[test]
fn test_dense_row_inside_a_vector_block_still_splits() {
    let n = 30;
    let (arena, root, n_states) = build_dense_row_vector_block_expr(n);
    let n_rows = CompiledExpr::new(&arena, root).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    // The dense row is element 0 of a len-2 matmul block: no scalar sub-node
    // exists for it, so it is synthesised by pushing the index into the matmul.
    assert_eq!(jac.n_dense_rows(), 1, "vector-block dense row must split");
    assert_eq!(jac.dense_rows()[0].rows, vec![n]);
    assert_eq!(jac.n_candidate_rows(), 1);
    assert!(
        jac.coloring().n_colors < n,
        "the split must lower the colour count, got {}",
        jac.coloring().n_colors
    );

    // Masking the row out of the parent tape needs a width-1 tangent node, which
    // a row inside a vector block has none of, so pruning is lost but not sense.
    assert!(
        std::ptr::eq(
            std::sync::Arc::as_ptr(jac.action_tape()),
            std::sync::Arc::as_ptr(jac.assembly_tape())
        ),
        "an unmaskable split must fall back to the unpruned tape"
    );
    // No colour may scatter into the split row -- `layout_in` asserts that for
    // every layout it builds, and building one here is what exercises it.
    let _ = jac.layout();

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 0.3).sin()).collect();
    fd_check_jacobian_data(&arena, root, &jac, &y);
}

#[test]
fn test_two_dense_rows_match_fd() {
    // Two wide rows over a sparse diagonal remainder; only the nonlinear one
    // is worth a reverse pass, since the linear one is known at compile time.
    let mut arena = Arena::new();
    let n = 20;
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let ones = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .unwrap(),
    )));
    let weights = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            (0..n).map(|i| 1.0 + i as f64).collect(),
            Shape::matrix(1, n),
        )
        .unwrap(),
    )));
    let sq = arena.alloc(Node::Mul(y, y));
    let dense0 = arena.alloc(Node::MatMul(ones, sq)); // sum y_i^2
    let dense1 = arena.alloc(Node::MatMul(weights, y)); // sum (1+i) y_i
    let mut rows = vec![dense0, dense1];
    for i in 2..n {
        let yi = arena.alloc(Node::Index {
            child: y,
            start: i,
            end: i + 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        rows.push(arena.alloc(Node::Mul(two, yi))); // sparse diagonal remainder
    }
    let root = arena.alloc(Node::Concat(rows));
    let n_rows = CompiledExpr::new(&arena, root).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
    assert_eq!(jac.n_dense_rows(), 1, "only the nonlinear row is split out");
    assert_eq!(jac.dense_rows()[0].rows[0], 0);
    let row1 = jac.sparsity().indptr[1]..jac.sparsity().indptr[2];
    assert_eq!(
        jac.constant_csr_entries()
            .iter()
            .filter(|&&(csr_idx, _)| row1.contains(&csr_idx))
            .count(),
        n,
        "the linear row comes wholly from the constant table"
    );

    let yv: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 0.3)).collect();
    fd_check_jacobian_data(&arena, root, &jac, &yv);

    // With the split off, both wide rows fall back to reverse mode.
    let reference = JacobianData::new_wrt_states_unsplit(&arena, root, n_rows, n);
    assert_eq!(reference.n_dense_rows(), 2);
    fd_check_jacobian_data(&arena, root, &reference, &yv);
}
