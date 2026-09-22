//! Solving many input sets concurrently.
//!
//! A fan-out over [`PreparedSolver::solve`], so a batch is bit-identical to the serial loop.
//! Runs on the ambient rayon pool; wrap the call in `ThreadPool::install` to set its width.

use rayon::prelude::*;

use super::solve::{InputSet, PreparedSolver, SolveOutcome, SolveRequest};
use crate::error::CoreError;

impl PreparedSolver {
    /// Solve one trajectory per input set, concurrently, each factorising serially.
    ///
    /// A set that fails keeps its own `Err` at its own index.
    pub fn solve_batch(
        &self,
        request: SolveRequest<'_>,
        sets: &[InputSet<'_>],
    ) -> Vec<Result<SolveOutcome, CoreError>> {
        // The sets already occupy the pool; a parallel LU in each would oversubscribe it.
        let request = request.with_lu_threads(1);
        sets.par_iter()
            .map(|set| self.solve(request, *set))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::test_fixtures::decay_with_output_and_event;

    /// The event at `y = 0.4` exercises root-finding across sets.
    fn prepared() -> PreparedSolver {
        PreparedSolver::new(decay_with_output_and_event(), 1e-8, &[1e-10]).expect("setup failed")
    }

    fn t_eval() -> Vec<f64> {
        (0..=40).map(|i| f64::from(i) * 0.05).collect()
    }

    /// Descending solve cost: the first set decays slowest.
    fn rates(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![0.4 + 0.35 * (i as f64)]).collect()
    }

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

    // Bit-identical is the property under test.
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
        assert_eq!(
            got.termination, want.termination,
            "{context}: terminations differ"
        );
        assert_eq!(got.y_event, want.y_event, "{context}: root states differ");
        assert_eq!(
            got.statistics.solver.number_of_steps, want.statistics.solver.number_of_steps,
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

    #[test]
    fn a_parallel_lu_solve_is_bit_identical_to_the_serial_one() {
        let solver = prepared();
        let times = t_eval();
        let rates = rates(1);
        let y0 = [1.0];
        let set = sets(&y0, &rates)[0];
        let request = SolveRequest::new(&times);

        let serial = solver.solve(request, set).expect("solve failed");
        let parallel =
            run_in_pool(4, || solver.solve(request.with_lu_threads(4), set)).expect("solve failed");
        assert_same_outcome(&parallel, &serial, "4 LU threads");
    }

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
    fn one_failing_set_leaves_the_others_intact() {
        let solver = prepared();
        let times = t_eval();
        let good = [1.0];
        let y0 = [1.0];
        // Set 2 gets the wrong input width: a failure independent of the integrator.
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
}
