//! Allocation-free faer sparse-LU linear solver.
//!
//! diffsol's stock `FaerSparseLU` allocates on every linearisation refresh (a
//! clone of the symbolic structure, a fresh `NumericLu`, and its factorisation
//! workspace) and on every triangular solve (the solve scratch, hundreds of
//! times per solve inside Newton). This solver keeps the symbolic
//! factorisation, the numeric storage, and both workspaces alive for its whole
//! lifetime, so a refresh runs only the numeric kernel and a Newton iteration
//! runs only the triangular kernels.

use std::cell::RefCell;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{FaerContext, LaError, Matrix, MatrixCommon, Vector};
use diffsol_la::error::LinearSolverError;
use diffsol_la::{LinearOp, LinearSolver};
use dyn_stack::{MemBuffer, MemStack};
use faer::reborrow::Reborrow;
use faer::sparse::linalg::lu::{
    LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu,
};
use faer::{Par, Spec};

/// Problem sizes here are a few thousand states at most, where rayon dispatch
/// costs more than it saves.
const PAR: Par = Par::Seq;

/// A [`LinearSolver`] over faer's sparse LU that reuses every buffer.
pub struct ReusedFaerLu {
    symbolic: Option<SymbolicLu<usize>>,
    numeric: NumericLu<usize, f64>,
    matrix: Option<FaerSparseMat<f64>>,
    factorize_buf: Option<MemBuffer>,
    /// `solve_in_place` takes `&self`, so its scratch hides behind a `RefCell`.
    solve_buf: RefCell<Option<MemBuffer>>,
    factorized: bool,
}

impl Default for ReusedFaerLu {
    fn default() -> Self {
        Self {
            symbolic: None,
            numeric: NumericLu::new(),
            matrix: None,
            factorize_buf: None,
            solve_buf: RefCell::new(None),
            factorized: false,
        }
    }
}

impl std::fmt::Debug for ReusedFaerLu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReusedFaerLu")
            .field("factorized", &self.factorized)
            .finish_non_exhaustive()
    }
}

impl LinearSolver<FaerSparseMat<f64>> for ReusedFaerLu {
    fn set_sparsity<
        C: LinearOp<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let matrix = FaerSparseMat::<f64>::new_from_sparsity(
            op.nrows(),
            op.ncols(),
            op.sparsity(),
            *op.context(),
        );
        let symbolic =
            factorize_symbolic_lu(matrix.inner().symbolic(), LuSymbolicParams::default())
                .expect("Failed to create symbolic LU");
        self.factorize_buf = Some(MemBuffer::new(
            symbolic.factorize_numeric_lu_scratch::<f64>(PAR, Spec::default()),
        ));
        self.solve_buf = RefCell::new(Some(MemBuffer::new(
            symbolic.solve_in_place_scratch::<f64>(1, PAR),
        )));
        self.symbolic = Some(symbolic);
        self.matrix = Some(matrix);
        self.factorized = false;
    }

    fn set_linearisation<
        C: LinearOp<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        let symbolic = self.symbolic.as_ref().expect("Sparsity not set");
        let stack = MemStack::new(self.factorize_buf.as_mut().expect("Sparsity not set"));
        symbolic
            .factorize_numeric_lu(
                &mut self.numeric,
                matrix.inner().rb(),
                PAR,
                stack,
                Spec::default(),
            )
            .expect("Failed to factorise matrix");
        self.factorized = true;
    }

    fn solve_in_place(&self, x: &mut FaerVec<f64>) -> Result<(), LaError> {
        if !self.factorized {
            return Err(LinearSolverError::LuNotInitialized.into());
        }
        let symbolic = self.symbolic.as_ref().expect("factorized implies symbolic");
        let lu = LuRef::new_unchecked(symbolic, &self.numeric);
        let mut buf = self.solve_buf.borrow_mut();
        let stack = MemStack::new(buf.as_mut().expect("factorized implies solve scratch"));
        lu.solve_in_place_with_conj(faer::Conj::No, x.inner_mut().as_mat_mut(), PAR, stack);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{Context, VectorHost};

    /// The pattern is asymmetric so the solve exercises both permutations.
    struct TestOp {
        matrix: FaerSparseMat<f64>,
        context: FaerContext,
    }

    impl TestOp {
        fn new() -> Self {
            let context = FaerContext::default();
            let indices = vec![(0, 0), (0, 2), (1, 1), (2, 1), (2, 2)];
            let values = vec![2.0, 1.0, 3.0, 4.0, 5.0];
            let matrix = FaerSparseMat::try_from_triplets(3, 3, indices, values, context)
                .expect("bad triplets");
            Self { matrix, context }
        }
    }

    impl LinearOp for TestOp {
        type T = f64;
        type V = FaerVec<f64>;
        type M = FaerSparseMat<f64>;
        type C = FaerContext;

        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            3
        }
        fn context(&self) -> &FaerContext {
            &self.context
        }
        fn matrix_inplace(&self, y: &mut FaerSparseMat<f64>) {
            y.copy_from(&self.matrix);
        }
        fn sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
            self.matrix.sparsity().map(|s| s.to_owned().unwrap())
        }
    }

    #[test]
    fn matches_stock_faer_sparse_lu() {
        let op = TestOp::new();

        let mut reused = ReusedFaerLu::default();
        reused.set_sparsity(&op);
        reused.set_linearisation(&op);

        let mut stock = diffsol::FaerSparseLU::<f64>::default();
        stock.set_sparsity(&op);
        stock.set_linearisation(&op);

        let ctx = FaerContext::default();
        let mut x_reused = ctx.vector_zeros::<FaerVec<f64>>(3);
        x_reused.as_mut_slice().copy_from_slice(&[5.0, 6.0, 23.0]);
        let mut x_stock = x_reused.clone();

        LinearSolver::solve_in_place(&reused, &mut x_reused).unwrap();
        LinearSolver::solve_in_place(&stock, &mut x_stock).unwrap();

        // Same factorisation algorithm, same arithmetic: bitwise equality.
        assert_eq!(x_reused.as_slice(), x_stock.as_slice());
        // A x = [5, 6, 23] with the matrix above has x = [1, 2, 3].
        for (got, want) in x_reused.as_slice().iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn refactorisation_reuses_buffers_and_stays_correct() {
        let op = TestOp::new();
        let mut solver = ReusedFaerLu::default();
        solver.set_sparsity(&op);

        let ctx = FaerContext::default();
        for _ in 0..3 {
            solver.set_linearisation(&op);
            let mut x = ctx.vector_zeros::<FaerVec<f64>>(3);
            x.as_mut_slice().copy_from_slice(&[5.0, 6.0, 23.0]);
            LinearSolver::solve_in_place(&solver, &mut x).unwrap();
            for (got, want) in x.as_slice().iter().zip(&[1.0, 2.0, 3.0]) {
                assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
            }
        }
    }

    #[test]
    fn solve_before_linearisation_is_an_error() {
        let op = TestOp::new();
        let mut solver = ReusedFaerLu::default();
        solver.set_sparsity(&op);
        let ctx = FaerContext::default();
        let mut x = ctx.vector_zeros::<FaerVec<f64>>(3);
        assert!(LinearSolver::solve_in_place(&solver, &mut x).is_err());
    }

    /// An op over an arbitrary square CSC matrix built from triplets.
    struct TripletOp {
        matrix: FaerSparseMat<f64>,
        context: FaerContext,
        n: usize,
    }

    impl TripletOp {
        fn new(n: usize, triplets: &[(usize, usize, f64)]) -> Self {
            let context = FaerContext::default();
            let indices: Vec<(usize, usize)> = triplets.iter().map(|&(r, c, _)| (r, c)).collect();
            let values: Vec<f64> = triplets.iter().map(|&(.., v)| v).collect();
            let matrix = FaerSparseMat::try_from_triplets(n, n, indices, values, context)
                .expect("bad triplets");
            Self { matrix, context, n }
        }
    }

    impl LinearOp for TripletOp {
        type T = f64;
        type V = FaerVec<f64>;
        type M = FaerSparseMat<f64>;
        type C = FaerContext;

        fn nrows(&self) -> usize {
            self.n
        }
        fn ncols(&self) -> usize {
            self.n
        }
        fn context(&self) -> &FaerContext {
            &self.context
        }
        fn matrix_inplace(&self, y: &mut FaerSparseMat<f64>) {
            y.copy_from(&self.matrix);
        }
        fn sparsity(&self) -> Option<<FaerSparseMat<f64> as Matrix>::Sparsity> {
            self.matrix.sparsity().map(|s| s.to_owned().unwrap())
        }
    }

    #[test]
    fn set_sparsity_again_rebuilds_for_a_new_pattern_and_size() {
        // One solver instance must survive a sparsity change: the symbolic
        // factorisation, matrix, and both scratch buffers are all size-bound.
        let mut solver = ReusedFaerLu::default();
        let ctx = FaerContext::default();

        let op3 = TestOp::new();
        solver.set_sparsity(&op3);
        solver.set_linearisation(&op3);
        let mut x = ctx.vector_zeros::<FaerVec<f64>>(3);
        x.as_mut_slice().copy_from_slice(&[5.0, 6.0, 23.0]);
        LinearSolver::solve_in_place(&solver, &mut x).unwrap();

        // 4x4 diagonal: a different pattern, dimension, and pivot structure.
        let op4 = TripletOp::new(4, &[(0, 0, 2.0), (1, 1, 4.0), (2, 2, 8.0), (3, 3, 16.0)]);
        solver.set_sparsity(&op4);
        // The old factorisation must not survive the sparsity change.
        let mut stale = ctx.vector_zeros::<FaerVec<f64>>(4);
        assert!(LinearSolver::solve_in_place(&solver, &mut stale).is_err());

        solver.set_linearisation(&op4);
        let mut x = ctx.vector_zeros::<FaerVec<f64>>(4);
        x.as_mut_slice().copy_from_slice(&[2.0, 4.0, 8.0, 16.0]);
        LinearSolver::solve_in_place(&solver, &mut x).unwrap();
        for (got, want) in x.as_slice().iter().zip(&[1.0, 1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn dae_structured_matrix_matches_stock() {
        // A DAE-shaped iteration matrix: tridiagonal differential block plus
        // two non-diagonally-dominant algebraic rows, so pivoting must permute.
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
            (2, 3, -1.0),
            (3, 3, 1e-3),
            (3, 4, 1.0),
            (4, 0, 1.0),
            (4, 4, -1.0),
        ];
        let op = TripletOp::new(5, &triplets);

        let mut reused = ReusedFaerLu::default();
        reused.set_sparsity(&op);
        let mut stock = diffsol::FaerSparseLU::<f64>::default();
        stock.set_sparsity(&op);

        let ctx = FaerContext::default();
        // Refactorise repeatedly, as Newton does, and compare every solve.
        for k in 0..3 {
            reused.set_linearisation(&op);
            stock.set_linearisation(&op);
            let mut b = ctx.vector_zeros::<FaerVec<f64>>(5);
            let rhs: Vec<f64> = (0..5).map(|i| f64::from(k * 3 + i + 1)).collect();
            b.as_mut_slice().copy_from_slice(&rhs);
            let mut b_stock = b.clone();
            LinearSolver::solve_in_place(&reused, &mut b).unwrap();
            LinearSolver::solve_in_place(&stock, &mut b_stock).unwrap();
            assert_eq!(b.as_slice(), b_stock.as_slice());
        }
    }

    #[test]
    fn singular_matrix_behaviour_matches_stock() {
        // Newton can hit a singular iteration matrix mid-solve; panic or
        // poison, the reused solver must match stock so diffsol recovers alike.
        let singular = || TripletOp::new(2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]);

        let reused_outcome = std::panic::catch_unwind(|| {
            let op = singular();
            let mut solver = ReusedFaerLu::default();
            solver.set_sparsity(&op);
            solver.set_linearisation(&op);
            let ctx = FaerContext::default();
            let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
            x.as_mut_slice().copy_from_slice(&[1.0, 2.0]);
            LinearSolver::solve_in_place(&solver, &mut x).unwrap();
            x.as_slice()
                .iter()
                .map(|v| v.is_finite())
                .collect::<Vec<_>>()
        });
        let stock_outcome = std::panic::catch_unwind(|| {
            let op = singular();
            let mut solver = diffsol::FaerSparseLU::<f64>::default();
            solver.set_sparsity(&op);
            solver.set_linearisation(&op);
            let ctx = FaerContext::default();
            let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
            x.as_mut_slice().copy_from_slice(&[1.0, 2.0]);
            LinearSolver::solve_in_place(&solver, &mut x).unwrap();
            x.as_slice()
                .iter()
                .map(|v| v.is_finite())
                .collect::<Vec<_>>()
        });

        match (reused_outcome, stock_outcome) {
            (Ok(reused), Ok(stock)) => assert_eq!(reused, stock),
            (Err(_), Err(_)) => {},
            (reused, stock) => panic!(
                "singular-matrix behaviour diverged from stock: reused ok={}, stock ok={}",
                reused.is_ok(),
                stock.is_ok()
            ),
        }
    }
}
