//! Solving many input sets concurrently.
//!
//! The one method here is a fan-out over [`PreparedSolver::solve`], never a
//! second implementation of a solve. That is what makes a batch bit-identical to
//! the serial loop by construction: one immutable tape shared through the `Arc`,
//! one fresh `Workspace` minted inside each `solve()`, and rayon only choosing
//! the order the independent calls run in.
//!
//! The [`SolveRequest`] is shared across the batch rather than per set, which
//! matches the callers: `DiffsolSolver` computes one output grid for every set,
//! `BaseSolver` refuses to run sets whose discontinuities differ, and the
//! payload flags come from the model. Only [`InputSet`] varies per set.
//!
//! Scheduling is the caller's: these run on the ambient rayon pool, so a caller
//! that wants a specific width wraps the call in `ThreadPool::install`.

use rayon::prelude::*;

use super::solve::{InputSet, PreparedSolver, SolveOutcome, SolveRequest};
use crate::error::CoreError;

/// Check that every per-set argument agrees on the batch width.
///
/// A `&[InputSet]` cannot itself disagree, so this is for callers assembling the
/// sets from separate untyped columns — the FFI boundary, where a `y0` array and
/// an `inputs` array arrive with independent row counts. `y0_sens` is `None` when
/// no seeds were supplied.
pub const fn check_batch_widths(
    y0: usize,
    inputs: usize,
    y0_sens: Option<usize>,
) -> Result<(), CoreError> {
    if y0 != inputs {
        return Err(CoreError::BatchWidths { y0, inputs });
    }
    if let Some(got) = y0_sens
        && got != y0
    {
        return Err(CoreError::BatchSensWidth { got, expected: y0 });
    }
    Ok(())
}

impl PreparedSolver {
    /// Solve one trajectory per input set, concurrently.
    ///
    /// Every set answers the same `request`, so the batch axis costs one method
    /// rather than one per payload combination. A set that fails to integrate
    /// keeps its own `Err` at its own index, so a caller can report which sets
    /// failed rather than collapsing the batch into one error.
    pub fn solve_batch(
        &self,
        request: SolveRequest<'_>,
        sets: &[InputSet<'_>],
    ) -> Vec<Result<SolveOutcome, CoreError>> {
        sets.par_iter()
            .map(|set| self.solve(request, *set))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::model::{CompiledModelOptions, ModelEvaluator};
    use crate::node::{CsrData, Node, Shape};

    /// `dy/dt = -p * y` with an event at `y - 0.4`, so a set's trajectory ends
    /// either at the root or at the final time depending on its own `p`. The
    /// event makes the batch exercise the root-finding and wind-back path, the
    /// part of a solve most likely to leak state between sets if any did.
    /// Sensitivities are requested for `p`, so one fixture serves both the plain
    /// and the sensitivity request.
    fn build_decay_with_event() -> ModelEvaluator {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let rate = arena.alloc(Node::InputParameter {
            name: "rate".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_rate = arena.alloc(Node::Mul(neg, rate));
        let rhs = arena.alloc(Node::Mul(neg_rate, sv));

        let threshold = arena.alloc(Node::Scalar(0.4));
        let event = arena.alloc(Node::Sub(sv, threshold));

        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape { rows: 1, cols: 1 },
        };
        let mut model = ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            1,
            1,
            CompiledModelOptions::new().with_sensitivities(&[0]),
        );
        model.add_event(&arena, event);
        model
    }

    fn prepared() -> PreparedSolver {
        PreparedSolver::new(build_decay_with_event(), 1e-8, &[1e-10]).expect("setup failed")
    }

    fn t_eval() -> Vec<f64> {
        (0..=40).map(|i| f64::from(i) * 0.05).collect()
    }

    /// Descending solve cost: the first set decays slowest, so it reaches the
    /// event last and takes the most steps.
    fn rates(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![0.4 + 0.35 * (i as f64)]).collect()
    }

    /// One set per rate, all starting from the same state.
    fn sets<'a>(y0: &'a [f64], rates: &'a [Vec<f64>]) -> Vec<InputSet<'a>> {
        rates
            .iter()
            .map(|inputs| InputSet::new(y0, inputs))
            .collect()
    }

    fn run_in_pool<T: Send>(threads: usize, body: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool build failed")
            .install(body)
    }

    /// Every field a batch is expected to reproduce exactly from the serial
    /// solve of the same set, payload included.
    // Bit-identical is the property under test, so the comparisons are exact by
    // design rather than by omission.
    #[allow(clippy::float_cmp)]
    fn assert_same_outcome(got: &SolveOutcome, want: &SolveOutcome, context: &str) {
        assert_eq!(got.t, want.t, "{context}: times differ");
        assert_eq!(got.y, want.y, "{context}: trajectories differ");
        assert_eq!(got.n_rows, want.n_rows, "{context}: row counts differ");
        assert_eq!(got.yp, want.yp, "{context}: yp differs");
        assert_eq!(
            got.sensitivities, want.sensitivities,
            "{context}: sensitivities differ"
        );
        assert_eq!(got.flag, want.flag, "{context}: flags differ");
        assert_eq!(got.t_event, want.t_event, "{context}: root times differ");
        assert_eq!(got.y_event, want.y_event, "{context}: root states differ");
        assert_eq!(
            got.statistics.number_of_steps, want.statistics.number_of_steps,
            "{context}: step counts differ",
        );
    }

    #[test]
    fn batch_is_bit_identical_to_the_serial_loop() {
        let solver = prepared();
        let times = t_eval();
        let rates = rates(8);
        let y0 = [1.0];
        let sets = sets(&y0, &rates);
        let request = SolveRequest::new(&times);

        let serial: Vec<_> = sets
            .iter()
            .map(|set| solver.solve(request, *set).expect("solve failed"))
            .collect();

        for threads in [1, 2, 8] {
            let batched = run_in_pool(threads, || solver.solve_batch(request, &sets));
            assert_eq!(batched.len(), serial.len());
            for (i, (got, want)) in batched.into_iter().zip(&serial).enumerate() {
                let got = got.expect("set failed");
                assert_same_outcome(&got, want, &format!("set {i} at {threads} thread(s)"));
            }
        }
    }

    /// The batch axis is one method for every payload combination, so the
    /// sensitivity blocks have to survive the fan-out as exactly as the states
    /// do — the property the four separate batch methods each needed their own
    /// test for.
    #[test]
    fn a_sensitivity_batch_is_bit_identical_to_the_serial_loop() {
        let solver = prepared();
        let times = t_eval();
        let rates = rates(4);
        let y0 = [1.0];
        let seed = [0.25];
        let sets: Vec<InputSet<'_>> = rates
            .iter()
            .map(|inputs| InputSet::new(&y0, inputs).with_sens_seed(&seed))
            .collect();
        let request = SolveRequest::new(&times).with_sensitivities();

        let serial: Vec<_> = sets
            .iter()
            .map(|set| solver.solve(request, *set).expect("solve failed"))
            .collect();

        let batched = run_in_pool(4, || solver.solve_batch(request, &sets));
        for (i, (got, want)) in batched.into_iter().zip(&serial).enumerate() {
            let got = got.expect("set failed");
            assert!(got.sensitivities.is_some(), "set {i} dropped its blocks");
            assert_same_outcome(&got, want, &format!("sensitivity set {i}"));
        }
    }

    #[test]
    // Result ordering is the property under test; the values are compared
    // exactly for the same reason as above.
    #[allow(clippy::float_cmp)]
    fn results_follow_input_order_under_heterogeneous_cost() {
        let solver = prepared();
        let times = t_eval();
        // Widely spread rates, so a batch that returned completion order rather
        // than input order would reverse them.
        let rates: Vec<Vec<f64>> = vec![vec![0.2], vec![1.0], vec![3.0], vec![9.0]];
        let y0 = [1.0];
        let sets = sets(&y0, &rates);
        let request = SolveRequest::new(&times);

        let batched = run_in_pool(4, || solver.solve_batch(request, &sets));

        for (i, (got, set)) in batched.into_iter().zip(&sets).enumerate() {
            let got = got.expect("set failed");
            let want = solver.solve(request, *set).expect("reference solve failed");
            assert_eq!(got.t, want.t, "set {i} landed out of order");
            assert_eq!(got.y, want.y, "set {i} landed out of order");
        }
    }

    #[test]
    fn one_failing_set_leaves_the_others_intact() {
        let solver = prepared();
        let times = t_eval();
        let good = [1.0];
        let y0 = [1.0];
        // Set 2 is handed the wrong input width, the one per-set failure a test
        // can provoke without depending on how the integrator diverges.
        let sets = [
            InputSet::new(&y0, &good),
            InputSet::new(&y0, &good),
            InputSet::new(&y0, &[]),
            InputSet::new(&y0, &good),
            InputSet::new(&y0, &good),
        ];

        let batched = run_in_pool(4, || solver.solve_batch(SolveRequest::new(&times), &sets));

        assert_eq!(batched.len(), 5);
        for (i, result) in batched.into_iter().enumerate() {
            if i == 2 {
                assert!(
                    matches!(result, Err(CoreError::InputsLength { .. })),
                    "set 2 should carry its own error",
                );
            } else {
                assert!(result.is_ok(), "set {i} failed alongside set 2");
            }
        }
    }

    #[test]
    fn mismatched_batch_widths_are_rejected_before_any_solve() {
        let err = check_batch_widths(2, 1, None).expect_err("mismatched widths accepted");
        assert!(matches!(err, CoreError::BatchWidths { y0: 2, inputs: 1 }));

        let err = check_batch_widths(2, 2, Some(1)).expect_err("mismatched seed count accepted");
        assert!(matches!(
            err,
            CoreError::BatchSensWidth {
                got: 1,
                expected: 2
            }
        ));
    }
}
