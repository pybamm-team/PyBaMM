//! Model builders and dense-matrix helpers shared by the model's test modules.

use super::{AlgebraicBlockSpec, CompiledModel, CompiledModelOptions};
use crate::arena::{Arena, NodeId};
use crate::error::CoreError;
use crate::node::{CsrData, Node, Shape};
use crate::observable::ObservableKind;
use crate::params::{PackedWidth, ParamIdx};

/// Compile a fixture known to be valid, with `options`.
pub fn compile_with(
    arena: &Arena,
    rhs: NodeId,
    mass: CsrData,
    n_states: usize,
    input_width: usize,
    options: CompiledModelOptions<'_>,
) -> CompiledModel {
    CompiledModel::new(
        arena,
        rhs,
        mass,
        n_states,
        PackedWidth::new(input_width),
        options,
    )
    .expect("a valid fixture")
}

/// Compile a fixture known to be valid, without options.
pub fn compile(
    arena: &Arena,
    rhs: NodeId,
    mass: CsrData,
    n_states: usize,
    input_width: usize,
) -> CompiledModel {
    compile_with(
        arena,
        rhs,
        mass,
        n_states,
        input_width,
        CompiledModelOptions::default(),
    )
}

/// Compile with sensitivities on `sens`.
pub fn compile_with_sens(
    arena: &Arena,
    rhs: NodeId,
    mass: CsrData,
    n_states: usize,
    input_width: usize,
    sens: &[ParamIdx],
) -> CompiledModel {
    let options = CompiledModelOptions {
        sensitivities: sens,
        ..CompiledModelOptions::default()
    };
    compile_with(arena, rhs, mass, n_states, input_width, options)
}

/// `options` with only an algebraic block over `variable_indices`.
pub const fn algebraic_options(
    residual: NodeId,
    variable_indices: &[usize],
) -> CompiledModelOptions<'_> {
    CompiledModelOptions {
        sensitivities: &[],
        algebraic: Some(AlgebraicBlockSpec {
            residual,
            variable_indices,
        }),
    }
}

/// `2 * y` over `n` states with identity mass and no inputs.
pub fn doubling_model(n: usize) -> CompiledModel {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: n });
    let two = arena.alloc(Node::Scalar(2.0));
    let rhs = arena.alloc(Node::Mul(two, y));
    compile(&arena, rhs, identity_mass_matrix(n), n, 0)
}

/// `y' = y` on state 0 beside `g = [y0 + y1*y2, y1 + y2^2]` over algebraic states 1 and 2.
pub fn split_algebraic_model() -> CompiledModel {
    let mut arena = Arena::new();
    let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let y0 = element(&mut arena, y_full, 0);
    let y1 = element(&mut arena, y_full, 1);
    let y2 = element(&mut arena, y_full, 2);
    let alg0_mul = arena.alloc(Node::Mul(y1, y2));
    let alg0 = arena.alloc(Node::Add(y0, alg0_mul));
    let two = arena.alloc(Node::Scalar(2.0));
    let y2_sq = arena.alloc(Node::Pow(y2, two));
    let alg1 = arena.alloc(Node::Add(y1, y2_sq));
    let alg = arena.alloc(Node::Concat(vec![alg0, alg1]));
    let mass = CsrData {
        shape: Shape::matrix(3, 3),
        indptr: vec![0, 1, 1, 1],
        indices: vec![0],
        data: vec![1.0],
    };

    compile_with(&arena, y_full, mass, 3, 0, algebraic_options(alg, &[1, 2]))
}

/// Element `i` of the vector `of`.
pub fn element(arena: &mut Arena, of: NodeId, i: usize) -> NodeId {
    arena.alloc(Node::Index {
        child: of,
        start: i,
        end: i + 1,
    })
}

pub fn input(arena: &mut Arena, name: &str, index: usize, offset: usize, width: usize) -> NodeId {
    arena.alloc(Node::InputParameter {
        name: name.to_string(),
        index,
        offset,
        width,
    })
}

pub fn identity_mass_matrix(n: usize) -> CsrData {
    CsrData {
        shape: Shape::matrix(n, n),
        indptr: (0..=n).collect(),
        indices: (0..n).collect(),
        data: vec![1.0; n],
    }
}

/// A mass matrix with no entries, so every state is algebraic.
pub fn all_algebraic_mass_matrix(n: usize) -> CsrData {
    CsrData {
        shape: Shape::matrix(n, n),
        indptr: vec![0; n + 1],
        indices: Vec::new(),
        data: Vec::new(),
    }
}

/// `CompiledModel::new` without options, for the rejection tests.
pub fn try_compile(
    arena: &Arena,
    rhs: NodeId,
    mass: CsrData,
    n_states: usize,
    input_width: usize,
) -> Result<CompiledModel, CoreError> {
    CompiledModel::new(
        arena,
        rhs,
        mass,
        n_states,
        PackedWidth::new(input_width),
        CompiledModelOptions::default(),
    )
}

/// `f = k0 * y + k1 * y^2`, `H = k0 * y[0] + k1 * y[1]`, with sensitivities for k1 alone.
pub fn build_trailing_param_sens_model() -> (CompiledModel, [f64; 2]) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let y0 = element(&mut arena, y, 0);
    let y1 = element(&mut arena, y, 1);
    let k0 = input(&mut arena, "k0", 0, 0, 1);
    let k1 = input(&mut arena, "k1", 1, 1, 1);
    let two = arena.alloc(Node::Scalar(2.0));
    let y_squared = arena.alloc(Node::Pow(y, two));
    let linear = arena.alloc(Node::Mul(k0, y));
    let quadratic = arena.alloc(Node::Mul(k1, y_squared));
    let rhs = arena.alloc(Node::Add(linear, quadratic));

    let out_left = arena.alloc(Node::Mul(k0, y0));
    let out_right = arena.alloc(Node::Mul(k1, y1));
    let out_node = arena.alloc(Node::Add(out_left, out_right));

    let mut model = compile_with_sens(
        &arena,
        rhs,
        identity_mass_matrix(2),
        2,
        2,
        &[ParamIdx::new(1)],
    );
    model
        .set_observables(ObservableKind::Outputs, &mut arena, &[out_node])
        .expect("a valid output");
    (model, [3.0, 4.0])
}

/// `n` states: a dense row `sum_{j in dense} y_j^2` beside `sin(y_i)` rows, with `mass`.
///
/// The dense row comes first, with `sin` over `y[1..]`, or last, with `sin` over `y[..n-1]`.
pub fn dense_row_model(
    n: usize,
    dense: std::ops::Range<usize>,
    dense_row_first: bool,
    mass: CsrData,
) -> CompiledModel {
    let mut arena = Arena::new();
    let read = arena.alloc(Node::StateVector {
        start: dense.start,
        end: dense.end,
    });
    let squares = arena.alloc(Node::Mul(read, read));
    let width = dense.len();
    let ones = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
        indptr: vec![0, width],
        indices: (0..width).collect(),
        data: vec![1.0; width],
        shape: Shape::matrix(1, width),
    })));
    let dense_row = arena.alloc(Node::MatMul(ones, squares));
    let (start, end) = if dense_row_first { (1, n) } else { (0, n - 1) };
    let diagonal_states = arena.alloc(Node::StateVector { start, end });
    let diagonal = arena.alloc(Node::Sin(diagonal_states));
    let rows = if dense_row_first {
        vec![dense_row, diagonal]
    } else {
        vec![diagonal, dense_row]
    };
    let rhs = arena.alloc(Node::Concat(rows));
    compile(&arena, rhs, mass, n, 0)
}

/// Dense central finite differences of `eval_rhs`.
pub fn finite_difference_jacobian(model: &CompiledModel, y: &[f64], eps: f64) -> Vec<Vec<f64>> {
    let n = model.n_states();
    let mut ws = model.create_workspace();
    let (mut fp, mut fm) = (vec![0.0; n], vec![0.0; n]);
    let mut dense = vec![vec![0.0; n]; n];
    for col in 0..n {
        let (mut yp, mut ym) = (y.to_vec(), y.to_vec());
        yp[col] += eps;
        ym[col] -= eps;
        model.eval_rhs(&mut ws, 0.0, &yp, &[], &mut fp);
        model.eval_rhs(&mut ws, 0.0, &ym, &[], &mut fm);
        for row in 0..n {
            dense[row][col] = (fp[row] - fm[row]) / (2.0 * eps);
        }
    }
    dense
}

pub fn assert_matrices_match(assembled: &[Vec<f64>], expected: &[Vec<f64>], tol: f64, ctx: &str) {
    let n = expected.len();
    for row in 0..n {
        for col in 0..n {
            let err = (assembled[row][col] - expected[row][col]).abs();
            let scale = assembled[row][col]
                .abs()
                .max(expected[row][col].abs())
                .max(1e-12);
            assert!(
                err / scale < tol || err < tol,
                "{ctx} J({row},{col}): assembled={}, expected={}",
                assembled[row][col],
                expected[row][col]
            );
        }
    }
}

/// `df/dy - cj*M` at `(y, inputs)`, assembled through the merged CSC and densified.
pub fn assemble_dense(model: &CompiledModel, cj: f64, y: &[f64], inputs: &[f64]) -> Vec<Vec<f64>> {
    let mut values = vec![f64::NAN; model.nnz()];
    let mut ws = model.create_workspace();
    model.assemble_jacobian_csc_into(&mut ws, 0.0, y, inputs, cj, &mut values);
    csc_to_dense(model, &values)
}

/// `df/dy` alone, assembled without the mass post-pass and densified.
pub fn assemble_dense_no_mass(model: &CompiledModel, y: &[f64]) -> Vec<Vec<f64>> {
    let mut values = vec![f64::NAN; model.nnz()];
    let mut ws = model.create_workspace();
    model.assemble_jacobian_csc_no_mass(&mut ws, 0.0, y, &[], &mut values);
    csc_to_dense(model, &values)
}

pub fn csc_to_dense(model: &CompiledModel, csc_vals: &[f64]) -> Vec<Vec<f64>> {
    let (n, csc) = (model.n_states(), model.csc_sparsity());
    let mut dense = vec![vec![0.0; n]; n];
    for (col, span) in csc.colptr().windows(2).enumerate() {
        for k in span[0]..span[1] {
            dense[csc.rowind()[k]][col] = csc_vals[k];
        }
    }
    dense
}

/// `expected` less `cj` on the first `n_differential` diagonal entries.
pub fn less_cj_on_the_diagonal(
    mut expected: Vec<Vec<f64>>,
    cj: f64,
    n_differential: usize,
) -> Vec<Vec<f64>> {
    for (i, row) in expected.iter_mut().enumerate().take(n_differential) {
        row[i] -= cj;
    }
    expected
}
