//! In-process DAE solving through diffsol.
//!
//! diffsol drives a model through one operator trait per callback it needs, and
//! this module supplies them from a [`ModelEvaluator`](crate::ModelEvaluator):
//! `rhs` for `f(t, y; p)` and its Jacobian, `mass` for `M`, `init` for `y0`,
//! `root` for events, `output` for observed variables, and `reset` as a required
//! no-op. `equations` binds those into the `OdeEquations` diffsol consumes,
//! `solve` owns problem setup and the solve loop, and `batch` fans that loop out
//! over many input sets on rayon.
//!
//! The operators share a field vocabulary: `compiled` is the shared immutable
//! [`CompiledModel`](crate::model::CompiledModel), `ws` the solve-local
//! scratch, `inputs` this solve's parameter values, and `context` faer's
//! allocator handle. Because `ws` is per solve, one solve's scratch is never
//! visible to another.
//!
//! The functions here translate our sparsity patterns into faer's, which is the
//! matrix backend diffsol is instantiated with throughout.

pub mod batch;
pub mod equations;
pub mod init;
pub mod linear;
pub mod mass;
pub mod observable;
pub mod options;
pub mod reset;
pub mod rhs;
pub mod solve;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use faer::sparse::SymbolicSparseColMat;

use crate::jacobian::CscPattern;
use crate::node::CsrData;

pub use self::equations::Equations;
pub use self::options::SolverOptions;

/// Sparsity type of the faer sparse matrix diffsol is instantiated with.
pub type FaerSparsity = <FaerSparseMat<f64> as diffsol::Matrix>::Sparsity;

/// Reinterpret a CSC Jacobian pattern as faer's symbolic sparsity.
///
/// Both sides are column-major, so the index arrays transfer verbatim; faer's
/// checked constructor is what validates them.
pub fn csc_to_faer_sparsity(csc: &CscPattern) -> FaerSparsity {
    let col_ptrs: Vec<usize> = csc.colptr.clone();
    let row_indices: Vec<usize> = csc.rowind.clone();
    SymbolicSparseColMat::new_checked(csc.nrows, csc.ncols, col_ptrs, None, row_indices)
}

/// Build a dense CSC sparsity for an `nrows × ncols` matrix.
///
/// Used for the `df/dp` sensitivity matrix where every state may depend on
/// every parameter.
pub fn dense_faer_sparsity(nrows: usize, ncols: usize) -> FaerSparsity {
    let col_ptrs: Vec<usize> = (0..=ncols).map(|c| c * nrows).collect();
    let row_indices: Vec<usize> = (0..ncols).flat_map(|_| 0..nrows).collect();
    SymbolicSparseColMat::new_checked(nrows, ncols, col_ptrs, None, row_indices)
}

/// Transpose a CSR mass matrix into faer's column-major sparsity plus the
/// value array in that CSC order.
///
/// The mass matrix arrives from Python in CSR, and its pattern is not assumed
/// symmetric, so the entries are redistributed by counting sort rather than
/// reinterpreted in place.
pub fn csr_mass_to_faer_csc(mass: &CsrData) -> (FaerSparsity, Vec<f64>) {
    let nrows = mass.shape.rows;
    let ncols = mass.shape.cols;

    let mut col_counts = vec![0usize; ncols + 1];
    for row in 0..nrows {
        for idx in mass.indptr[row]..mass.indptr[row + 1] {
            col_counts[mass.indices[idx] + 1] += 1;
        }
    }
    for c in 1..=ncols {
        col_counts[c] += col_counts[c - 1];
    }

    let nnz = mass.indptr[nrows];
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    let mut current = col_counts.clone();
    for row in 0..nrows {
        for idx in mass.indptr[row]..mass.indptr[row + 1] {
            let col = mass.indices[idx];
            row_indices[current[col]] = row;
            values[current[col]] = mass.data[idx];
            current[col] += 1;
        }
    }

    let sparsity = SymbolicSparseColMat::new_checked(nrows, ncols, col_counts, None, row_indices);
    (sparsity, values)
}

/// Transpose a CSR mass matrix's pattern into faer's column-major sparsity.
pub fn csr_mass_to_faer_sparsity(mass: &CsrData) -> FaerSparsity {
    csr_mass_to_faer_csc(mass).0
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use super::init::InitOp;
    use super::mass::MassOp;
    use super::rhs::RhsOp;
    use super::solve::{InputSet, SolveRequest};
    use super::*;
    use crate::arena::Arena;
    use crate::model::{CompiledModel, ModelEvaluator, Workspace};
    use crate::node::{CsrData, Node, Shape};
    use diffsol::vector::faer_serial::FaerVec;
    use diffsol::{
        ConstantOp, Context, FaerContext, LinearOp, Matrix, NonLinearOp, NonLinearOpJacobian, Op,
        VectorHost,
    };

    /// Owns what a borrowed operator view points at, as `Equations` does in a
    /// real solve, so one op can be exercised on its own.
    struct OpFixture {
        compiled: Arc<CompiledModel>,
        ws: RefCell<Workspace>,
        jac_sparsity: FaerSparsity,
        mass_sparsity: FaerSparsity,
        mass_csc_values: Vec<f64>,
        inputs: Vec<f64>,
        sens_params: Vec<usize>,
        n_states: usize,
        context: FaerContext,
    }

    impl OpFixture {
        fn new(model: ModelEvaluator) -> Self {
            let compiled = model.into_compiled();
            let ws = RefCell::new(compiled.create_workspace());
            let jac_sparsity = csc_to_faer_sparsity(compiled.csc_sparsity());
            let (mass_sparsity, mass_csc_values) = csr_mass_to_faer_csc(compiled.mass_matrix());
            let n_params = compiled.n_params();
            Self {
                n_states: compiled.n_states(),
                compiled,
                ws,
                jac_sparsity,
                mass_sparsity,
                mass_csc_values,
                inputs: vec![0.0; n_params],
                sens_params: (0..n_params).collect(),
                context: FaerContext::default(),
            }
        }

        fn rhs(&self) -> RhsOp<'_> {
            RhsOp {
                compiled: &self.compiled,
                ws: &self.ws,
                inputs: &self.inputs,
                sens_params: &self.sens_params,
                jac_sparsity: &self.jac_sparsity,
                n_states: self.n_states,
                context: self.context,
            }
        }

        fn mass(&self) -> MassOp<'_> {
            MassOp {
                compiled: &self.compiled,
                ws: &self.ws,
                sparsity: &self.mass_sparsity,
                csc_values: &self.mass_csc_values,
                n_states: self.n_states,
                context: self.context,
            }
        }
    }

    fn build_2state_model() -> ModelEvaluator {
        // dy0/dt = -y0, dy1/dt = -2*y1
        // Jacobian = [[-1, 0], [0, -2]]
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let neg_two = arena.alloc(Node::Scalar(-2.0));
        let rhs0 = arena.alloc(Node::Mul(neg_one, sv0));
        let rhs1 = arena.alloc(Node::Mul(neg_two, sv1));
        let rhs = arena.alloc(Node::Concat(vec![rhs0, rhs1]));

        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 1.0],
            shape: Shape { rows: 2, cols: 2 },
        };

        ModelEvaluator::new(&arena, rhs, mass, 2, 0)
    }

    fn build_1state_model() -> ModelEvaluator {
        // dy/dt = -y
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs_expr = arena.alloc(Node::Mul(neg, sv));

        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape { rows: 1, cols: 1 },
        };

        ModelEvaluator::new(&arena, rhs_expr, mass, 1, 0)
    }

    #[test]
    fn csc_to_faer_preserves_pattern() {
        let model = build_2state_model();
        let csc = model.csc_sparsity();
        let sparsity = csc_to_faer_sparsity(csc);

        assert_eq!(sparsity.nrows(), csc.nrows);
        assert_eq!(sparsity.ncols(), csc.ncols);
        assert_eq!(sparsity.row_idx().len(), csc.rowind.len());

        // Verify col_ptrs match exactly
        let faer_col_ptrs = sparsity.col_ptr();
        assert_eq!(
            faer_col_ptrs.len(),
            csc.colptr.len(),
            "col_ptrs length mismatch"
        );
        for (i, (&faer_val, &csc_val)) in faer_col_ptrs.iter().zip(&csc.colptr).enumerate() {
            assert_eq!(faer_val, csc_val, "col_ptrs[{i}] mismatch");
        }

        // Verify row_indices match exactly
        let faer_row_idx = sparsity.row_idx();
        assert_eq!(
            faer_row_idx.len(),
            csc.rowind.len(),
            "row_idx length mismatch"
        );
        for (i, (&faer_val, &csc_val)) in faer_row_idx.iter().zip(&csc.rowind).enumerate() {
            assert_eq!(faer_val, csc_val, "row_idx[{i}] mismatch");
        }
    }

    #[test]
    fn csr_mass_to_faer_preserves_entries() {
        let model = build_2state_model();
        let mass = model.mass_matrix();
        let sparsity = csr_mass_to_faer_sparsity(mass);

        assert_eq!(sparsity.nrows(), mass.shape.rows);
        assert_eq!(sparsity.ncols(), mass.shape.cols);

        let nnz_csr = mass.indptr[mass.shape.rows];
        assert_eq!(
            sparsity.row_idx().len(),
            nnz_csr,
            "nnz mismatch between CSR and faer CSC"
        );

        // Collect (row, col) pairs from CSR
        let mut csr_entries: Vec<(usize, usize)> = Vec::new();
        for row in 0..mass.shape.rows {
            for idx in mass.indptr[row]..mass.indptr[row + 1] {
                csr_entries.push((row, mass.indices[idx]));
            }
        }
        csr_entries.sort_unstable();

        // Collect (row, col) pairs from faer CSC sparsity
        let col_ptrs = sparsity.col_ptr();
        let row_idx = sparsity.row_idx();
        let mut faer_entries: Vec<(usize, usize)> = Vec::new();
        for col in 0..sparsity.ncols() {
            for &row in &row_idx[col_ptrs[col]..col_ptrs[col + 1]] {
                faer_entries.push((row, col));
            }
        }
        faer_entries.sort_unstable();

        assert_eq!(
            csr_entries, faer_entries,
            "structural entries differ between CSR source and faer CSC"
        );
    }

    #[test]
    fn solve_1state_exponential_decay() {
        // dy/dt = -y, y(0) = 1. Exact: y(t) = exp(-t)
        let model = build_1state_model();

        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let result =
            solve::solve(model, &t_eval, &[], &[1.0], &[], 1e-8, &[1e-10]).expect("solve failed");

        assert_eq!(result.flag, 0, "should complete without events");
        assert!(result.t_event.is_none(), "no events expected");
        assert!(result.y_event.is_none(), "no event state expected");
        assert_eq!(result.n_rows, 1);
        assert!(
            result.n_times >= 10,
            "should have at least 10 output points, got {}",
            result.n_times
        );

        // Verify monotonically increasing time
        for w in result.t.windows(2) {
            assert!(w[1] > w[0], "time should be monotonically increasing");
        }

        // Verify accuracy against exact solution
        for (j, &t) in result.t.iter().enumerate() {
            let exact = (-t).exp();
            let y_val = result.y[j * result.n_rows];
            let error = (y_val - exact).abs();
            assert!(
                error < 1e-6,
                "t={t:.1}: y={y_val:.8}, exact={exact:.8}, error={error:.2e}",
            );
        }
    }

    #[test]
    fn solve_2state_independent_decay() {
        // dy0/dt = -y0, dy1/dt = -2*y1 from y = [1, 1], so the exact solution is
        // y0(t) = exp(-t), y1(t) = exp(-2t).
        let model = build_2state_model();

        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.05).collect();
        let result = solve::solve(model, &t_eval, &[], &[1.0, 1.0], &[], 1e-8, &[1e-10, 1e-10])
            .expect("solve failed");

        assert_eq!(result.flag, 0);

        for (j, &t) in result.t.iter().enumerate() {
            let y0 = result.y[j * result.n_rows];
            let y1 = result.y[j * result.n_rows + 1];

            let exact0 = (-t).exp();
            let exact1 = (-2.0 * t).exp();
            let err0 = (y0 - exact0).abs();
            let err1 = (y1 - exact1).abs();

            assert!(
                err0 < 1e-6,
                "t={t:.2}: y0={y0:.8}, exact={exact0:.8}, err={err0:.2e}",
            );
            assert!(
                err1 < 1e-6,
                "t={t:.2}: y1={y1:.8}, exact={exact1:.8}, err={err1:.2e}",
            );
        }
    }

    #[test]
    fn prepared_problem_repeated_solves_match() {
        let model = build_2state_model();
        let atol = vec![1e-10, 1e-10];
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.05).collect();

        // Reference: single solve through the free function
        let ref_result = solve::solve(
            build_2state_model(),
            &t_eval,
            &[],
            &[1.0, 1.0],
            &[],
            1e-8,
            &atol,
        )
        .expect("reference solve failed");

        // PreparedSolver: multiple solves on the same problem
        let prepared =
            solve::PreparedSolver::new(model, 1e-8, &atol).expect("PreparedSolver creation failed");

        let result1 = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0, 1.0], &[]))
            .expect("first prepared solve failed");
        let result2 = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0, 1.0], &[]))
            .expect("second prepared solve failed");

        // Both prepared solves should match the reference
        assert_eq!(result1.n_times, ref_result.n_times);
        assert_eq!(result2.n_times, ref_result.n_times);
        assert_eq!(result1.y.len(), ref_result.y.len(), "result lengths differ");
        for i in 0..result1.y.len() {
            assert!(
                (result1.y[i] - ref_result.y[i]).abs() < 1e-10,
                "result1.y[{i}] mismatch: {} vs {}",
                result1.y[i],
                ref_result.y[i]
            );
            assert!(
                (result2.y[i] - ref_result.y[i]).abs() < 1e-10,
                "result2.y[{i}] mismatch: {} vs {}",
                result2.y[i],
                ref_result.y[i]
            );
        }
    }

    fn build_1state_model_with_output() -> ModelEvaluator {
        // dy/dt = -y, output = 2*y
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs_expr = arena.alloc(Node::Mul(neg, sv));

        let two = arena.alloc(Node::Scalar(2.0));
        let output_expr = arena.alloc(Node::Mul(two, sv));

        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape { rows: 1, cols: 1 },
        };

        let mut model = ModelEvaluator::new(&arena, rhs_expr, mass, 1, 0);
        model.add_output(&arena, output_expr);
        model
    }

    #[test]
    fn prepared_problem_outputs_request_matches_full() {
        // dy/dt = -y, y(0) = 1. Output = 2*y.
        // Verify outputs = 2 * states from full solve.
        let model_full = build_1state_model_with_output();
        let model_out = build_1state_model_with_output();
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();

        // Full solve
        let prepared_full =
            solve::PreparedSolver::new(model_full, 1e-8, &atol).expect("full setup failed");
        let full_result = prepared_full
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .expect("full solve failed");

        // Output-only solve
        let prepared_out =
            solve::PreparedSolver::new(model_out, 1e-8, &atol).expect("output setup failed");
        let out_result = prepared_out
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&[1.0], &[]),
            )
            .expect("output solve failed");

        // Output = 2*y, so outputs should be 2x the state values
        assert_eq!(out_result.n_rows, 1);
        assert_eq!(out_result.n_times, full_result.n_times);
        for j in 0..out_result.n_times {
            let state = full_result.y[j * full_result.n_rows];
            let output = out_result.y[j * out_result.n_rows];
            let expected = 2.0 * state;
            assert!(
                (output - expected).abs() < 1e-8,
                "t={}: output={output}, expected={expected}",
                out_result.t[j]
            );
        }
    }

    #[test]
    fn prepared_problem_different_y0() {
        let model = build_1state_model();
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();

        let prepared =
            solve::PreparedSolver::new(model, 1e-8, &atol).expect("PreparedSolver creation failed");

        let result_a = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .expect("solve with y0=1 failed");
        let result_b = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[2.0], &[]))
            .expect("solve with y0=2 failed");

        // y(t) = y0 * exp(-t), so result_b should be 2x result_a
        for (j, &t) in result_a.t.iter().enumerate() {
            let ya = result_a.y[j];
            let yb = result_b.y[j];
            let exact_a = (-t).exp();
            let exact_b = 2.0 * (-t).exp();
            assert!(
                (ya - exact_a).abs() < 1e-6,
                "y0=1: t={t}, y={ya}, expected {exact_a}"
            );
            assert!(
                (yb - exact_b).abs() < 1e-6,
                "y0=2: t={t}, y={yb}, expected {exact_b}"
            );
        }
    }

    #[test]
    fn rhs_eval_2state_decay() {
        let fixture = OpFixture::new(build_2state_model());
        let rhs = fixture.rhs();
        let ctx = FaerContext::default();
        assert_eq!(rhs.nstates(), 2);
        assert_eq!(rhs.nout(), 2);

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        x.as_mut_slice().copy_from_slice(&[3.0, 5.0]);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(2);
        rhs.call_inplace(&x, 0.0, &mut y);

        // f = [-y0, -2*y1] => [-3, -10]
        assert!((y.as_slice()[0] - (-3.0)).abs() < 1e-12);
        assert!((y.as_slice()[1] - (-10.0)).abs() < 1e-12);
    }

    #[test]
    fn rhs_jac_mul_2state() {
        let fixture = OpFixture::new(build_2state_model());
        let rhs = fixture.rhs();
        let ctx = FaerContext::default();

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        x.as_mut_slice().copy_from_slice(&[1.0, 1.0]);
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(2);
        v.as_mut_slice().copy_from_slice(&[2.0, 3.0]);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(2);

        rhs.jac_mul_inplace(&x, 0.0, &v, &mut y);

        // df/dy = diag(-1, -2), so (df/dy) @ v = [-v0, -2*v1] = [-2, -6]
        assert!((y.as_slice()[0] - (-2.0)).abs() < 1e-12);
        assert!((y.as_slice()[1] - (-6.0)).abs() < 1e-12);
    }

    #[test]
    fn rhs_jacobian_inplace_2state() {
        let fixture = OpFixture::new(build_2state_model());
        let rhs = fixture.rhs();
        let ctx = FaerContext::default();

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        x.as_mut_slice().copy_from_slice(&[1.0, 1.0]);

        // Allocate the sparse matrix from the op's own sparsity and assemble via
        // jacobian_inplace (the diffsol `jacobian` default does exactly this).
        let jac = rhs.jacobian(&x, 0.0);

        // df/dy = diag(-1, -2). Collect the assembled triplets and check.
        let mut diag = [0.0f64; 2];
        let (indices, values) = jac.triplet_iter();
        for ((row, col), val) in indices.zip(values) {
            assert_eq!(row, col, "jacobian must be diagonal, got ({row},{col})");
            diag[row] = val;
        }
        assert!((diag[0] - (-1.0)).abs() < 1e-12, "J[0,0]={}", diag[0]);
        assert!((diag[1] - (-2.0)).abs() < 1e-12, "J[1,1]={}", diag[1]);
    }

    #[test]
    fn mass_gemv_identity_beta_zero() {
        let fixture = OpFixture::new(build_2state_model());
        let mass = fixture.mass();
        let ctx = FaerContext::default();

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        x.as_mut_slice().copy_from_slice(&[7.0, -4.0]);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(2);

        // beta == 0: y = M @ x. Mass is identity => y == x.
        mass.gemv_inplace(&x, 0.0, 0.0, &mut y);
        assert!((y.as_slice()[0] - 7.0).abs() < 1e-12);
        assert!((y.as_slice()[1] - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn mass_gemv_identity_beta_nonzero() {
        let fixture = OpFixture::new(build_2state_model());
        let mass = fixture.mass();
        let ctx = FaerContext::default();

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        x.as_mut_slice().copy_from_slice(&[1.0, 2.0]);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(2);
        y.as_mut_slice().copy_from_slice(&[10.0, 20.0]);

        // beta == 3: y = M @ x + beta * y_old = x + 3*y_old
        // = [1 + 30, 2 + 60] = [31, 62]
        mass.gemv_inplace(&x, 0.0, 3.0, &mut y);
        assert!(
            (y.as_slice()[0] - 31.0).abs() < 1e-12,
            "y0={}",
            y.as_slice()[0]
        );
        assert!(
            (y.as_slice()[1] - 62.0).abs() < 1e-12,
            "y1={}",
            y.as_slice()[1]
        );
    }

    #[test]
    fn mass_has_sparsity() {
        let fixture = OpFixture::new(build_2state_model());
        assert!(fixture.mass().sparsity().is_some());
    }

    #[test]
    fn mass_matrix_inplace_matches_probing_default() {
        // The copy-based override must reproduce diffsol's column-probing
        // default entry for entry, on identity and DAE (zero-row) masses.
        for model in [build_2state_model(), build_dae_model()] {
            let fixture = OpFixture::new(model);
            let mass = fixture.mass();
            let n = fixture.n_states;
            let ctx = FaerContext::default();
            let mut fast = FaerSparseMat::<f64>::new_from_sparsity(n, n, mass.sparsity(), ctx);
            let mut probed = FaerSparseMat::<f64>::new_from_sparsity(n, n, mass.sparsity(), ctx);
            mass.matrix_inplace(0.0, &mut fast);
            mass._default_matrix_inplace(0.0, &mut probed);

            let (fast_idx, fast_vals) = fast.triplet_iter();
            let (probed_idx, probed_vals) = probed.triplet_iter();
            let fast_entries: Vec<_> = fast_idx.zip(fast_vals).collect();
            let probed_entries: Vec<_> = probed_idx.zip(probed_vals).collect();
            assert_eq!(fast_entries, probed_entries);
        }
    }

    #[test]
    fn init_copies_y0_exactly() {
        let ctx = FaerContext::default();
        let init = InitOp {
            y0: &[1.5, -2.5, 3.0],
            y0_sens: &[],
            n_states: 3,
            n_sens_params: 0,
            context: ctx,
        };
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(3);
        init.call_inplace(0.0, &mut y);
        assert_eq!(y.as_slice(), &[1.5, -2.5, 3.0]);
    }

    #[test]
    fn sequential_solves_are_isolated() {
        // Same PreparedSolver, two solves with different y0. The fresh
        // Workspace-per-solve must keep them fully independent.
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared = solve::PreparedSolver::new(build_1state_model(), 1e-8, &atol).unwrap();

        let a = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .unwrap();
        let b = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[5.0], &[]))
            .unwrap();

        for (j, &t) in a.t.iter().enumerate() {
            let exact = (-t).exp();
            assert!(
                (a.y[j * a.n_rows] - exact).abs() < 1e-6,
                "a t={t}: {} vs {exact}",
                a.y[j * a.n_rows]
            );
            assert!(
                5.0f64.mul_add(-exact, b.y[j * b.n_rows]).abs() < 1e-6,
                "b t={t}: {} vs {}",
                b.y[j * b.n_rows],
                5.0 * exact
            );
        }
    }

    #[test]
    fn solve_takes_shared_ref() {
        // `prepared` is intentionally NOT `mut`; `solve` takes `&self`.
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.1).collect();
        let prepared = solve::PreparedSolver::new(build_1state_model(), 1e-8, &atol).unwrap();
        let r1 = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .unwrap();
        let r2 = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .unwrap();
        assert_eq!(r1.n_times, r2.n_times);
    }

    #[test]
    fn prepared_problem_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<solve::PreparedSolver>();
    }

    #[test]
    fn an_outputs_request_matches_analytic() {
        // dy/dt = -y, y(0) = 1, output = 2*y => 2*exp(-t).
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared =
            solve::PreparedSolver::new(build_1state_model_with_output(), 1e-8, &atol).unwrap();
        let r = prepared
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&[1.0], &[]),
            )
            .unwrap();

        assert_eq!(r.n_rows, 1);
        for (j, &t) in r.t.iter().enumerate() {
            let expected = 2.0 * (-t).exp();
            let out = r.y[j * r.n_rows];
            assert!(
                (out - expected).abs() < 1e-6,
                "t={t}: output={out}, expected={expected}"
            );
        }
    }

    #[test]
    fn an_outputs_request_at_final_time_carries_the_terminal_full_state() {
        // Without a state trajectory, y_event is the only state a caller can
        // restart from, so it must be present on final-time termination too.
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared_full =
            solve::PreparedSolver::new(build_1state_model_with_output(), 1e-8, &atol).unwrap();
        let full = prepared_full
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .unwrap();
        let prepared_out =
            solve::PreparedSolver::new(build_1state_model_with_output(), 1e-8, &atol).unwrap();
        let r = prepared_out
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&[1.0], &[]),
            )
            .unwrap();

        assert_eq!(r.flag, 0);
        assert!(r.t_event.is_none());
        let y_event = r.y_event.expect("terminal state missing");
        assert_eq!(y_event.len(), 1);
        let full_last = full.y[(full.n_times - 1) * full.n_rows];
        assert!(
            (y_event[0] - full_last).abs() < 1e-8,
            "terminal state {} diverges from full solve {full_last}",
            y_event[0]
        );
    }

    fn build_decay_model_with_output_and_event() -> ModelEvaluator {
        // dy/dt = -y, y(0) = 1, output = 2*y, event = y - 0.5 (root at t = ln 2).
        // Output and state differ there, so y_event tells them apart.
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs_expr = arena.alloc(Node::Mul(neg, sv));

        let two = arena.alloc(Node::Scalar(2.0));
        let output_expr = arena.alloc(Node::Mul(two, sv));

        let half = arena.alloc(Node::Scalar(0.5));
        let event_expr = arena.alloc(Node::Sub(sv, half));

        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape { rows: 1, cols: 1 },
        };

        let mut model = ModelEvaluator::new(&arena, rhs_expr, mass, 1, 0);
        model.add_output(&arena, output_expr);
        model.add_event(&arena, event_expr);
        model
    }

    #[test]
    // The trajectory's final time is the event time stored verbatim, so the
    // assertion checks for a bit-identical value, not an approximate one.
    #[allow(clippy::float_cmp)]
    fn solve_event_ends_trajectory_at_root_with_full_state() {
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared =
            solve::PreparedSolver::new(build_decay_model_with_output_and_event(), 1e-8, &atol)
                .unwrap();
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[]))
            .unwrap();

        let ln2 = std::f64::consts::LN_2;
        assert_eq!(r.flag, 1, "expected event termination");
        let t_event = r.t_event.expect("t_event missing");
        assert!(
            (t_event - ln2).abs() < 1e-6,
            "t_event={t_event}, expected {ln2}"
        );
        // The trajectory includes the root time and state as its final column.
        assert_eq!(r.n_times, r.t.len());
        assert_eq!(*r.t.last().unwrap(), t_event);
        let y_last = r.y[(r.n_times - 1) * r.n_rows];
        assert!(
            (y_last - 0.5).abs() < 1e-6,
            "y at event={y_last}, expected 0.5"
        );
        // y_event is the full state at the root.
        let y_event = r.y_event.expect("y_event missing");
        assert_eq!(y_event.len(), 1);
        assert!(
            (y_event[0] - 0.5).abs() < 1e-6,
            "y_event={}, expected state 0.5",
            y_event[0]
        );
    }

    #[test]
    // The trajectory's final time is the event time stored verbatim, so the
    // assertion checks for a bit-identical value, not an approximate one.
    #[allow(clippy::float_cmp)]
    fn an_outputs_request_returns_a_full_state_y_event_on_an_event() {
        let atol = vec![1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared =
            solve::PreparedSolver::new(build_decay_model_with_output_and_event(), 1e-8, &atol)
                .unwrap();
        let r = prepared
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&[1.0], &[]),
            )
            .unwrap();

        let ln2 = std::f64::consts::LN_2;
        assert_eq!(r.flag, 1, "expected event termination");
        let t_event = r.t_event.expect("t_event missing");
        assert!(
            (t_event - ln2).abs() < 1e-6,
            "t_event={t_event}, expected {ln2}"
        );
        // The trajectory includes the root time and output value as its final column.
        assert_eq!(r.n_times, r.t.len());
        assert_eq!(*r.t.last().unwrap(), t_event);
        let out_last = r.y[(r.n_times - 1) * r.n_rows];
        assert!(
            (out_last - 1.0).abs() < 1e-6,
            "output at event={out_last}, expected 1.0"
        );
        // y_event is the full state at the root, not the outputs row.
        let y_event = r.y_event.expect("y_event missing");
        assert_eq!(y_event.len(), 1);
        assert!(
            (y_event[0] - 0.5).abs() < 1e-6,
            "y_event={}, expected state 0.5, not output 1.0",
            y_event[0]
        );
    }

    fn build_dae_model() -> ModelEvaluator {
        // Differential: y0' = -y0.  Algebraic: 0 = 2*y0 - y1  (=> y1 = 2*y0).
        // Mass = diag(1, 0).
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg1 = arena.alloc(Node::Scalar(-1.0));
        let two = arena.alloc(Node::Scalar(2.0));
        let r0 = arena.alloc(Node::Mul(neg1, sv0));
        let two_y0 = arena.alloc(Node::Mul(two, sv0));
        let r1 = arena.alloc(Node::Sub(two_y0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![r0, r1]));
        let mass = CsrData {
            indptr: vec![0, 1, 1], // row0: (0,0)=1; row1: empty => 0
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(2, 2),
        };
        ModelEvaluator::new(&arena, rhs, mass, 2, 0)
    }

    #[test]
    fn dae_consistent_ic_and_solve() {
        let atol = vec![1e-10, 1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared = solve::PreparedSolver::new(build_dae_model(), 1e-8, &atol).unwrap();
        // y0=[1.0, 0.0]: algebraic y1 START is deliberately INCONSISTENT (should be 2.0).
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0, 0.0], &[]))
            .unwrap();
        assert_eq!(r.flag, 0);
        for (j, &t) in r.t.iter().enumerate() {
            let y0 = r.y[j * r.n_rows];
            let y1 = r.y[j * r.n_rows + 1];
            assert!((y0 - (-t).exp()).abs() < 1e-6, "t={t}: y0={y0}");
            assert!(
                2.0f64.mul_add(-(-t).exp(), y1).abs() < 1e-6,
                "t={t}: y1={y1} violates y1=2*y0"
            );
        }
    }
}
