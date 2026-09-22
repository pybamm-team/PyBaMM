//! In-process DAE solving through diffsol.
//!
//! diffsol applies `RhsOp::jac_mul_inplace` and `MassOp::gemv_inplace` to `S`, one lane per
//! parameter; every other callback expects one lane, read with `lane` or `lane_mut`.

pub mod batch;
mod equations;
mod init;
mod mass;
mod observable;
pub mod options;
mod rhs;
pub mod solve;
#[cfg(test)]
mod test_fixtures;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{Context, Vector, VectorCommon};
use faer::sparse::{SparseRowMatRef, SymbolicSparseColMat, SymbolicSparseRowMatRef};

use crate::jacobian::CscPattern;
use crate::node::CsrData;

pub use self::options::SolverOptions;

/// Sparsity type of the faer sparse matrix diffsol is instantiated with.
pub(crate) type FaerSparsity = <FaerSparseMat<f64> as diffsol::Matrix>::Sparsity;

/// Borrow the values of a single-lane vector.
pub(crate) fn lane(v: &FaerVec<f64>) -> &[f64] {
    assert_single_lane(v.context().nbatch());
    v.inner().col_as_slice(0)
}

/// Mutably borrow the values of a single-lane vector.
pub(crate) fn lane_mut(v: &mut FaerVec<f64>) -> &mut [f64] {
    assert_single_lane(v.context().nbatch());
    v.inner_mut().col_as_slice_mut(0)
}

/// Borrow lane `j` of a batched vector.
#[cfg(test)]
pub(crate) fn lane_at(v: &FaerVec<f64>, j: usize) -> &[f64] {
    v.inner().col_as_slice(j)
}

/// Iterate over the lanes of a batched vector.
pub(crate) fn lanes(v: &FaerVec<f64>) -> impl ExactSizeIterator<Item = &[f64]> {
    let values = v.inner();
    (0..values.ncols()).map(move |j| values.col_as_slice(j))
}

/// Iterate mutably over the lanes of a batched vector.
pub(crate) fn lanes_mut(v: &mut FaerVec<f64>) -> impl ExactSizeIterator<Item = &mut [f64]> {
    v.inner_mut().col_iter_mut().map(|column| {
        column
            .try_as_col_major_mut()
            .expect("a faer vector lane is one contiguous column")
            .as_slice_mut()
    })
}

/// Mutably borrow the stored values of a single-lane sparse matrix, in CSC order.
pub(crate) fn values_mut(m: &mut FaerSparseMat<f64>) -> &mut [f64] {
    let lanes = diffsol::Matrix::inner_mut(m);
    assert_single_lane(lanes.len());
    lanes[0].val_mut()
}

fn assert_single_lane(nbatch: usize) {
    assert_eq!(
        nbatch, 1,
        "a single-lane callback was handed {nbatch} lanes"
    );
}

/// Reinterpret a CSC Jacobian pattern as faer's symbolic sparsity.
pub(crate) fn csc_to_faer_sparsity(csc: &CscPattern) -> FaerSparsity {
    let col_ptrs: Vec<usize> = csc.colptr().to_vec();
    let row_indices: Vec<usize> = csc.rowind().to_vec();
    SymbolicSparseColMat::new_checked(csc.nrows(), csc.ncols(), col_ptrs, None, row_indices)
}

/// Build a dense CSC sparsity for an `nrows × ncols` matrix.
pub(crate) fn dense_faer_sparsity(nrows: usize, ncols: usize) -> FaerSparsity {
    let col_ptrs: Vec<usize> = (0..=ncols).map(|c| c * nrows).collect();
    let row_indices: Vec<usize> = (0..ncols).flat_map(|_| 0..nrows).collect();
    SymbolicSparseColMat::new_checked(nrows, ncols, col_ptrs, None, row_indices)
}

/// Transpose a CSR mass matrix into faer's CSC sparsity plus the values in that order.
pub(crate) fn csr_mass_to_faer_csc(mass: &CsrData) -> (FaerSparsity, Vec<f64>) {
    let shape = mass.shape();
    let symbolic = SymbolicSparseRowMatRef::new_unsorted_checked(
        shape.rows,
        shape.cols,
        mass.indptr(),
        None,
        mass.indices(),
    );
    SparseRowMatRef::new(symbolic, mass.data())
        .to_col_major()
        .expect("transposing M fails only to allocate")
        .into_parts()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use super::init::InitOp;
    use super::mass::MassOp;
    use super::rhs::RhsOp;
    use super::solve::{InputSet, PreparedSolver, SolveRequest, Termination};
    use super::test_fixtures::{
        decay, decay_with_output_and_event, differential_then_algebraic_mass,
    };
    use super::*;
    use crate::arena::Arena;
    use crate::model::test_fixtures::{compile, identity_mass_matrix};
    use crate::model::{CompiledModel, Workspace};
    use crate::node::{CsrData, Node, Shape};
    use crate::params::ParamBinding;
    use diffsol::vector::faer_serial::FaerVec;
    use diffsol::{
        ConstantOp, Context, FaerContext, LinearOp, Matrix, NonLinearOp, NonLinearOpJacobian, Op,
    };

    /// Owns what a borrowed operator view points at, so one op can be exercised alone.
    struct OpFixture {
        compiled: Arc<CompiledModel>,
        ws: RefCell<Workspace>,
        jac_sparsity: FaerSparsity,
        mass_sparsity: FaerSparsity,
        mass_csc_values: Vec<f64>,
        inputs: Vec<f64>,
        binding: ParamBinding,
        n_states: usize,
        context: FaerContext,
    }

    impl OpFixture {
        fn new(model: CompiledModel) -> Self {
            let compiled = Arc::new(model);
            let ws = RefCell::new(compiled.create_workspace());
            let jac_sparsity = csc_to_faer_sparsity(compiled.csc_sparsity());
            let (mass_sparsity, mass_csc_values) = csr_mass_to_faer_csc(compiled.mass_matrix());
            let width = compiled.input_width();
            let binding = ParamBinding::packed_inputs(width);
            Self {
                n_states: compiled.n_states(),
                compiled,
                ws,
                jac_sparsity,
                mass_sparsity,
                mass_csc_values,
                inputs: vec![0.0; width.get()],
                binding,
                context: FaerContext::default(),
            }
        }

        fn rhs(&self) -> RhsOp<'_> {
            RhsOp {
                compiled: &self.compiled,
                ws: &self.ws,
                inputs: &self.inputs,
                binding: &self.binding,
                jac_sparsity: &self.jac_sparsity,
                context: self.context,
            }
        }

        fn mass(&self) -> MassOp<'_> {
            MassOp {
                compiled: &self.compiled,
                sparsity: &self.mass_sparsity,
                csc_values: &self.mass_csc_values,
                context: self.context,
            }
        }
    }

    fn build_2state_model() -> CompiledModel {
        // dy0/dt = -y0, dy1/dt = -2*y1
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let neg_two = arena.alloc(Node::Scalar(-2.0));
        let rhs0 = arena.alloc(Node::Mul(neg_one, sv0));
        let rhs1 = arena.alloc(Node::Mul(neg_two, sv1));
        let rhs = arena.alloc(Node::Concat(vec![rhs0, rhs1]));
        compile(&arena, rhs, identity_mass_matrix(2), 2, 0)
    }

    #[test]
    fn csc_to_faer_preserves_pattern() {
        let model = build_2state_model();
        let csc = model.csc_sparsity();
        let sparsity = csc_to_faer_sparsity(csc);
        assert_eq!(sparsity.col_ptr(), csc.colptr());
        assert_eq!(sparsity.row_idx(), csc.rowind());
    }

    #[test]
    // Values are moved, never computed.
    #[allow(clippy::float_cmp)]
    fn csr_mass_to_faer_moves_each_value_to_its_column() {
        // Row 1 lists its columns out of order; each CSC column still runs down the rows.
        let mass = CsrData {
            indptr: vec![0, 2, 4, 5],
            indices: vec![0, 2, 2, 1, 0],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            shape: Shape::matrix(3, 3),
        };
        let (sparsity, values) = csr_mass_to_faer_csc(&mass);
        assert_eq!(sparsity.col_ptr(), [0, 2, 3, 5]);
        assert_eq!(sparsity.row_idx(), [0, 2, 1, 0, 1]);
        assert_eq!(values, [1.0, 5.0, 4.0, 2.0, 3.0]);
    }

    #[test]
    fn a_two_state_decay_matches_the_analytic_solution() {
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.05).collect();
        let result = PreparedSolver::new(build_2state_model(), 1e-8, &[1e-10, 1e-10])
            .expect("prepare")
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0, 1.0], &[]))
            .expect("solve");

        assert_eq!(result.termination, Termination::Completed);
        assert!(result.y_event.is_none(), "no event state expected");
        assert_eq!(result.t, t_eval);
        for (j, &t) in result.t.iter().enumerate() {
            let exact = [(-t).exp(), (-2.0 * t).exp()];
            for (row, exact) in exact.into_iter().enumerate() {
                let got = result.y[j * result.n_rows + row];
                assert!(
                    (got - exact).abs() < 1e-6,
                    "y{row}({t}) = {got}, exact {exact}"
                );
            }
        }
    }

    #[test]
    // Repeating an input set must repeat its trajectory bit for bit.
    #[allow(clippy::float_cmp)]
    fn one_prepared_solver_keeps_its_solves_independent() {
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared = PreparedSolver::new(decay(), 1e-8, &[1e-10]).expect("prepare");
        let [first, scaled, again] = [1.0, 5.0, 1.0].map(|y0| {
            prepared
                .solve(SolveRequest::new(&t_eval), InputSet::new(&[y0], &[]))
                .expect("solve")
        });

        assert_eq!(first.t, again.t);
        assert_eq!(first.y, again.y);
        for (&t, &y) in scaled.t.iter().zip(&scaled.y) {
            let exact = 5.0 * (-t).exp();
            assert!((y - exact).abs() < 5e-6, "y({t}) = {y}, exact {exact}");
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
        lane_mut(&mut x).copy_from_slice(&[3.0, 5.0]);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(2);
        rhs.call_inplace(&x, 0.0, &mut y);

        assert!((lane(&y)[0] - (-3.0)).abs() < 1e-12);
        assert!((lane(&y)[1] - (-10.0)).abs() < 1e-12);
    }

    fn build_2state_nonlinear_model() -> CompiledModel {
        // dy0/dt = -y0 * y1, dy1/dt = y0 * y0
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let product = arena.alloc(Node::Mul(sv0, sv1));
        let rhs0 = arena.alloc(Node::Mul(neg_one, product));
        let rhs1 = arena.alloc(Node::Mul(sv0, sv0));
        let rhs = arena.alloc(Node::Concat(vec![rhs0, rhs1]));
        compile(&arena, rhs, identity_mass_matrix(2), 2, 0)
    }

    fn batched(lanes: &[&[f64]]) -> FaerVec<f64> {
        let ctx = FaerContext::with_nbatch(lanes.len());
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(lanes[0].len());
        for (j, values) in lanes.iter().enumerate() {
            v.inner_mut().col_as_slice_mut(j).copy_from_slice(values);
        }
        v
    }

    #[test]
    fn rhs_jac_mul_takes_every_lane_at_one_state() {
        // Lane counts cover the scalar path, a padded batch, a full batch and a tail.
        let fixture = OpFixture::new(build_2state_nonlinear_model());
        let rhs = fixture.rhs();
        let state = [0.7, -1.3];
        let mut x = FaerContext::default().vector_zeros::<FaerVec<f64>>(2);
        lane_mut(&mut x).copy_from_slice(&state);

        for n_lanes in [1usize, 2, 3, 8, 9] {
            let directions: Vec<[f64; 2]> = (0..n_lanes)
                .map(|j| {
                    let j = j as f64;
                    [j - 1.5, 0.25 * j * j + 1.0]
                })
                .collect();
            let as_slices: Vec<&[f64]> = directions.iter().map(<[f64; 2]>::as_slice).collect();
            let v = batched(&as_slices);
            let zeros: &[f64] = &[0.0; 2];
            let mut y = batched(&vec![zeros; n_lanes]);
            rhs.jac_mul_inplace(&x, 0.0, &v, &mut y);

            for (j, direction) in directions.iter().enumerate() {
                let mut expected = [0.0; 2];
                fixture.compiled.jac_action(
                    &mut fixture.ws.borrow_mut(),
                    0.0,
                    &state,
                    &fixture.inputs,
                    direction,
                    &mut expected,
                );
                assert_eq!(lane_at(&y, j), expected, "lane {j} of {n_lanes}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "a single-lane callback was handed 2 lanes")]
    fn a_single_lane_callback_rejects_a_batched_state() {
        let fixture = OpFixture::new(build_2state_model());
        let x = batched(&[&[1.0, 1.0], &[2.0, 2.0]]);
        let mut y = batched(&[&[0.0; 2], &[0.0; 2]]);
        fixture.rhs().call_inplace(&x, 0.0, &mut y);
    }

    #[test]
    fn rhs_jacobian_inplace_2state() {
        let fixture = OpFixture::new(build_2state_model());
        let rhs = fixture.rhs();
        let ctx = FaerContext::default();

        let mut x = ctx.vector_zeros::<FaerVec<f64>>(2);
        lane_mut(&mut x).copy_from_slice(&[1.0, 1.0]);

        let jac = rhs.jacobian(&x, 0.0);

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
    fn mass_gemv_applies_m_to_every_lane() {
        let fixture = OpFixture::new(build_dae_model());
        let mass = fixture.mass();
        let x = batched(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mut y = batched(&[&[10.0, 20.0], &[30.0, 40.0]]);

        mass.gemv_inplace(&x, 0.0, 3.0, &mut y);
        assert_eq!(lane_at(&y, 0), [31.0, 60.0]);
        assert_eq!(lane_at(&y, 1), [93.0, 120.0]);
    }

    #[test]
    fn mass_matrix_inplace_matches_probing_default() {
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
            n_sens_params: 0,
            context: ctx,
        };
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(3);
        init.call_inplace(0.0, &mut y);
        assert_eq!(lane(&y), &[1.5, -2.5, 3.0]);
    }

    #[test]
    fn prepared_problem_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PreparedSolver>();
    }

    #[test]
    fn an_outputs_request_at_final_time_carries_the_terminal_full_state() {
        // Without a trajectory, y_event is the only state a caller can restart from.
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.1).collect();
        let prepared = PreparedSolver::new(decay_with_output_and_event(), 1e-8, &[1e-10]).unwrap();
        let set = InputSet::new(&[1.0], &[1.0]);
        let full = prepared.solve(SolveRequest::new(&t_eval), set).unwrap();
        let r = prepared
            .solve(SolveRequest::new(&t_eval).with_outputs(), set)
            .unwrap();

        assert_eq!(r.termination, Termination::Completed);
        let y_event = r.y_event.expect("terminal state missing");
        assert_eq!(y_event.len(), 1);
        let full_last = full.y[(full.t.len() - 1) * full.n_rows];
        assert!(
            (y_event[0] - full_last).abs() < 1e-8,
            "terminal state {} diverges from full solve {full_last}",
            y_event[0]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // the root is reported at the first requested time itself
    fn an_event_non_positive_at_the_start_ends_the_solve_there() {
        // The event starts below zero and stays there: no sign change for root finding.
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared = PreparedSolver::new(decay_with_output_and_event(), 1e-8, &[1e-10]).unwrap();
        for request in [
            SolveRequest::new(&t_eval),
            SolveRequest::new(&t_eval).with_outputs(),
        ] {
            let r = prepared
                .solve(request, InputSet::new(&[0.3], &[1.0]))
                .unwrap();

            assert_eq!(r.termination, Termination::Event { t: 0.0 });
            assert_eq!(r.t, vec![0.0]);
            assert_eq!(r.y_event.expect("y_event missing"), vec![0.3]);
        }
    }

    fn build_dae_model() -> CompiledModel {
        // y0' = -y0 and algebraic 0 = 2*y0 - y1, with M = diag(1, 0).
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg1 = arena.alloc(Node::Scalar(-1.0));
        let two = arena.alloc(Node::Scalar(2.0));
        let r0 = arena.alloc(Node::Mul(neg1, sv0));
        let two_y0 = arena.alloc(Node::Mul(two, sv0));
        let r1 = arena.alloc(Node::Sub(two_y0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![r0, r1]));
        compile(&arena, rhs, differential_then_algebraic_mass(), 2, 0)
    }

    #[test]
    fn dae_consistent_ic_and_solve() {
        let atol = vec![1e-10, 1e-10];
        let t_eval: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
        let prepared = PreparedSolver::new(build_dae_model(), 1e-8, &atol).unwrap();
        // The algebraic y1 starts deliberately inconsistent (should be 2.0).
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0, 0.0], &[]))
            .unwrap();
        assert_eq!(r.termination, Termination::Completed);
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
