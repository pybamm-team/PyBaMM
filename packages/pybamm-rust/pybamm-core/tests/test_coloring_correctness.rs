//! Dense rows split out of the colouring, checked against the unsplit build and central
//! finite differences.

use pybamm_core::DagRef;
use pybamm_core::NodeId;
use pybamm_core::arena::Arena;
use pybamm_core::eval::CompiledExpr;
use pybamm_core::jacobian::{JacobianData, JacobianScratch};
use pybamm_core::node::{ArrayData, CsrData, InterpolantData, Node, Shape};

/// `sin(y)` concatenated with one dense scalar row `ones(1 x n) @ (y*y)`.
fn build_dense_row_expr(n: usize) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let elt = arena.alloc(Node::Sin(y));
    let gy = arena.alloc(Node::Mul(y, y));
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
    let root = arena.alloc(Node::Concat(vec![elt, dense]));
    (arena, root, n)
}

/// Dense row inside a `2 x n` matmul block, so the split has to synthesise a scalar node.
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
    let block = arena.alloc(Node::MatMul(a, gy));
    let root = arena.alloc(Node::Concat(vec![elt, block]));
    (arena, root, n)
}

/// Assemble `jac`'s CSC values at `t = 0.5`.
fn assemble_csc(jac: &JacobianData, y: &[f64]) -> Vec<f64> {
    let layout = jac.layout();
    let mut scratch = JacobianScratch::new(jac);
    let mut data = vec![0.0; layout.n_slots()];
    jac.assemble_into(&mut scratch, layout, 0.5, y, &[], &[], &mut data);
    data
}

/// Compare every entry of the assembled Jacobian, zeros off the pattern included,
/// against central finite differences.
fn fd_check_jacobian_data(arena: &Arena, root: NodeId, jac: &JacobianData, y: &[f64]) {
    let primal = CompiledExpr::new(DagRef::new(arena, root));
    let dense = assemble_dense(jac, y);

    let mut s = vec![0.0; primal.scratch_len()];
    let eps = 1e-6;
    for col in 0..jac.n_cols() {
        let mut yp = y.to_vec();
        yp[col] += eps;
        let mut ym = y.to_vec();
        ym[col] -= eps;
        let fp = primal.eval(&mut s, 0.5, &yp, &[], &[]).to_vec();
        let fm = primal.eval(&mut s, 0.5, &ym, &[], &[]).to_vec();
        for (row, assembled) in dense.iter().enumerate() {
            let (dk, fd) = (assembled[col], (fp[row] - fm[row]) / (2.0 * eps));
            assert!(
                (dk - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "entry ({row},{col}): assembled {dk} vs fd {fd}"
            );
        }
    }
}

/// A block of `width` fully dense rows plus a diagonal remainder, as a 2-D collector has.
fn build_dense_block_expr(
    n: usize,
    width: usize,
    coupled: impl FnOnce(&mut Arena, NodeId) -> NodeId,
) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let diagonal = arena.alloc(Node::Sin(y));
    let coupled = coupled(&mut arena, y);
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

/// Assemble `jac` into a dense matrix.
fn assemble_dense(jac: &JacobianData, y: &[f64]) -> Vec<Vec<f64>> {
    let data = assemble_csc(jac, y);
    let mut dense = vec![vec![0.0; jac.n_cols()]; jac.n_rows()];
    for (col, span) in jac.csc().colptr().windows(2).enumerate() {
        for k in span[0]..span[1] {
            dense[jac.csc().rowind()[k]][col] = data[k];
        }
    }
    dense
}

#[test]
fn test_dense_block_splits_every_row_and_matches_the_unsplit_build() {
    // Dense rows are a minority of the states: the regime where the split pays off.
    let (n, width) = (64, 16);
    let (arena, root, n_states) =
        build_dense_block_expr(n, width, |arena, y| arena.alloc(Node::Mul(y, y)));
    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    assert_eq!(jac.n_dense_rows(), width, "the whole plateau must split");
    assert_eq!(jac.n_candidate_rows(), width);
    assert!(
        jac.n_colors() <= width + 1,
        "the split must leave the sparse remainder cheap to colour, got {}",
        jac.n_colors()
    );

    // A full-column subset builds the same derivative with splitting off.
    let all: Vec<usize> = (0..n_states).collect();
    let reference = JacobianData::new_wrt_state_subset(&arena, root, n_rows, n_states, &all);
    assert_eq!(reference.n_dense_rows(), 0);
    assert!(reference.n_colors() > jac.n_colors());

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

#[test]
fn test_a_dense_block_through_an_interpolant_splits() {
    // An interpolant once declined the split, leaving a complete graph to colour.
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

    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);
    assert_eq!(jac.n_dense_rows(), width, "the whole plateau must split");
    assert_eq!(jac.n_candidate_rows(), width);

    let y_values: Vec<f64> = (0..n)
        .map(|i| (i as f64).mul_add(0.11, 0.2).sin())
        .collect();
    fd_check_jacobian_data(&arena, root, &jac, &y_values);
}

#[test]
fn test_dense_array_rows_with_zero_coefficients_match_fd() {
    // The split tape skips zero coefficients, so it never loads some columns of its rows.
    let n = 30;
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let diagonal = arena.alloc(Node::Sin(y));
    let squared = arena.alloc(Node::Mul(y, y));
    let coefficients: Vec<f64> = (0..2 * n)
        .map(|k| {
            if k % 3 == 0 {
                0.0
            } else {
                (k as f64).mul_add(0.05, 0.5)
            }
        })
        .collect();
    let matrix = arena.alloc(Node::Array(Box::new(
        ArrayData::try_new(coefficients, Shape::matrix(2, n)).expect("valid matrix"),
    )));
    let block = arena.alloc(Node::MatMul(matrix, squared));
    let root = arena.alloc(Node::Concat(vec![diagonal, block]));
    let jac = JacobianData::new_wrt_states(&arena, root, n + 2, n);
    assert_eq!(jac.n_dense_rows(), 2, "both dense rows must split");

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 0.3).sin()).collect();
    fd_check_jacobian_data(&arena, root, &jac, &y);
}

#[test]
fn test_dense_row_split_matches_fd() {
    let n = 30;
    let (arena, root, n_states) = build_dense_row_expr(n);
    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    assert_eq!(jac.n_dense_rows(), 1, "dense row must be split out");
    assert!(
        jac.n_colors() <= 4,
        "residual coloring must reflect the sparse structure, got {}",
        jac.n_colors()
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

#[test]
fn test_dense_row_inside_a_vector_block_still_splits() {
    let n = 30;
    let (arena, root, n_states) = build_dense_row_vector_block_expr(n);
    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

    assert_eq!(jac.n_dense_rows(), 1, "vector-block dense row must split");
    assert!(jac.dense_row_indices().eq([n]));
    assert_eq!(jac.n_candidate_rows(), 1);
    assert!(
        jac.n_colors() < n,
        "the split must lower the colour count, got {}",
        jac.n_colors()
    );

    // No width-1 tangent node to mask the row out of, so pruning is lost.
    assert!(
        std::ptr::eq(
            std::sync::Arc::as_ptr(jac.action_tape()),
            std::sync::Arc::as_ptr(jac.assembly_tape())
        ),
        "an unmaskable split must fall back to the unpruned tape"
    );
    // `layout_in` asserts no colour scatters into the split row.
    let _ = jac.layout();

    let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.1, 0.3).sin()).collect();
    fd_check_jacobian_data(&arena, root, &jac, &y);
}

#[test]
fn test_two_dense_rows_match_fd() {
    // Only the nonlinear wide row needs a reverse pass; the linear one is constant.
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
    let dense0 = arena.alloc(Node::MatMul(ones, sq));
    let dense1 = arena.alloc(Node::MatMul(weights, y));
    let mut rows = vec![dense0, dense1];
    for i in 2..n {
        let yi = arena.alloc(Node::Index {
            child: y,
            start: i,
            end: i + 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        rows.push(arena.alloc(Node::Mul(two, yi)));
    }
    let root = arena.alloc(Node::Concat(rows));
    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
    assert_eq!(jac.n_dense_rows(), 1, "only the nonlinear row is split out");
    assert_eq!(jac.dense_row_indices().next(), Some(0));
    let row1 = jac.sparsity().row_span(1);
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

    let reference = JacobianData::new_wrt_states_sweeping_constants(&arena, root, n_rows, n);
    assert_eq!(reference.n_dense_rows(), 2);
    fd_check_jacobian_data(&arena, root, &reference, &yv);
}
