//! Problem setup and the solve loop.
//!
//! [`PreparedSolver`] holds everything reusable about a model, meaning the
//! immutable compiled model, the converted sparsity patterns and the sizes, so
//! repeated solves pay setup once. Each solve then builds its own workspace, its
//! own diffsol `OdeSolverProblem` and its own diffsol instance, which is what
//! keeps sequential solves independent and the prepared handle `Send + Sync`.
//!
//! What a solve carries back — states or output variables, with or without
//! sensitivities — is asked for on [`SolveRequest`] and answered in one
//! [`SolveOutcome`], so the payload combinations share one trajectory layout and
//! one set of termination fields rather than a result type each.

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::ode_solver::OdeSolverStatistics;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    AugmentedOdeEquationsImplicit, AugmentedOdeSolverMethod, Context, FaerContext,
    NewtonNonlinearSolver, NoLineSearch, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens,
    OdeBuilder, OdeEquations, OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem,
    OdeSolverStopReason, StateRefMut, Vector, VectorHost,
};

use super::equations::Equations;
use super::linear::ReusedFaerLu;
use super::options::SolverOptions;
use super::{FaerSparsity, csc_to_faer_sparsity, csr_mass_to_faer_csc};
use crate::error::CoreError;
use crate::model::{CompiledModel, Workspace};
use crate::node::CsrData;

/// Default multiplier applied to `atol` on differential rows to form the
/// forward-sensitivity absolute tolerance floor.
pub const DEFAULT_SENS_ATOL_FACTOR: f64 = 1e-3;

/// Solver statistics from a BDF solve.
#[derive(Debug, Clone)]
pub struct SolverStatistics {
    /// Accepted BDF steps.
    pub number_of_steps: usize,
    /// Jacobian/LU setups in total; the `_from_*` fields below break down why.
    pub number_of_linear_solver_setups: usize,
    /// Newton iterations across all steps.
    pub number_of_nonlinear_solver_iterations: usize,
    /// Newton solves that failed to converge.
    pub number_of_nonlinear_solver_fails: usize,
    /// Steps rejected by the local error test.
    pub number_of_error_test_failures: usize,
    /// Jacobian/LU setups triggered by checkpoint or reinitialisation.
    pub number_of_linear_solver_setups_from_checkpoint: usize,
    /// Jacobian/LU setups triggered by a first nonlinear convergence failure.
    pub number_of_linear_solver_setups_from_first_convergence_fail: usize,
    /// Jacobian/LU setups triggered by a second nonlinear convergence failure.
    pub number_of_linear_solver_setups_from_second_convergence_fail: usize,
    /// Jacobian/LU setups triggered by a local error test failure.
    pub number_of_linear_solver_setups_from_error_test_fail: usize,
    /// Jacobian/LU setups triggered by the normal step-success heuristic.
    pub number_of_linear_solver_setups_from_step_success: usize,
    /// Time spent computing consistent initial conditions (seconds).
    pub ic_time_secs: f64,
    /// Time spent creating the BDF solver instance (seconds).
    pub solver_setup_time_secs: f64,
    /// Wall-clock time for the whole solve (seconds), covering the two phases
    /// above and the integration itself but no FFI marshalling. Measured here
    /// rather than by the caller because a batched solve has no caller-side
    /// moment that corresponds to one set's integration.
    pub integration_time_secs: f64,
    /// True when the sensitivity solve failed under error control and was
    /// retried with sensitivities excluded from it.
    pub sens_error_control_relaxed: bool,
}

impl From<&OdeSolverStatistics> for SolverStatistics {
    fn from(stats: &OdeSolverStatistics) -> Self {
        Self {
            number_of_steps: stats.number_of_steps,
            number_of_linear_solver_setups: stats.number_of_linear_solver_setups,
            number_of_nonlinear_solver_iterations: stats.number_of_nonlinear_solver_iterations,
            number_of_nonlinear_solver_fails: stats.number_of_nonlinear_solver_fails,
            number_of_error_test_failures: stats.number_of_error_test_failures,
            number_of_linear_solver_setups_from_checkpoint: stats
                .number_of_linear_solver_setups_from_checkpoint,
            number_of_linear_solver_setups_from_first_convergence_fail: stats
                .number_of_linear_solver_setups_from_first_convergence_fail,
            number_of_linear_solver_setups_from_second_convergence_fail: stats
                .number_of_linear_solver_setups_from_second_convergence_fail,
            number_of_linear_solver_setups_from_error_test_fail: stats
                .number_of_linear_solver_setups_from_error_test_fail,
            number_of_linear_solver_setups_from_step_success: stats
                .number_of_linear_solver_setups_from_step_success,
            ic_time_secs: 0.0,
            solver_setup_time_secs: 0.0,
            integration_time_secs: 0.0,
            sens_error_control_relaxed: false,
        }
    }
}

/// What one solve returns, whichever payloads were asked for.
///
/// The termination fields live here once rather than once per payload
/// combination. `flag` is 0 when the requested time span completed and 1 when an
/// event root stopped the solve; on a root, `t` ends at the root time rather
/// than at the last requested point and `t_event` repeats it.
#[derive(Debug)]
pub struct SolveOutcome {
    /// Output times, one per trajectory column.
    pub t: Vec<f64>,
    /// Flat trajectory in column-major order: `y[i + j * n_rows]` = row i at
    /// time j. Rows are states, or the model's output variables when
    /// [`SolveRequest::outputs`] asked for them.
    pub y: Vec<f64>,
    /// Rows in `y`, and in each sensitivity block.
    pub n_rows: usize,
    /// Columns in `y`, equal to `t.len()`.
    pub n_times: usize,
    /// Row time derivatives sharing `y`'s layout. Present only on a state
    /// trajectory, when [`PreparedSolver::with_store_yp`] is set and the grid is
    /// under [`MAX_HERMITE_COLUMNS`]: each column is the derivative of the BDF
    /// interpolating polynomial at `t[j]`, the knot slope cubic-Hermite output
    /// interpolation needs.
    pub yp: Option<Vec<f64>>,
    /// One flat block per requested sensitivity parameter, sharing `y`'s layout,
    /// or `None` when the request asked for none. Blocks follow the order the
    /// sensitivity parameters were requested in, not global parameter order. On
    /// an outputs request each block already carries the full derivative
    /// `dg/dp + dg/dy · y_s`, not the `dg/dy` term alone.
    pub sensitivities: Option<Vec<Vec<f64>>>,
    /// Root time, `None` unless an event stopped the solve.
    pub t_event: Option<f64>,
    /// The full state where the trajectory ends, never an outputs row: on a root
    /// the state at the root time, and on an outputs request the terminal state
    /// as well, the only one such a caller can restart from.
    pub y_event: Option<Vec<f64>>,
    /// 0 for a completed span, 1 for an event root.
    pub flag: i32,
    /// Step and setup counters from the diffsol run, including whether
    /// sensitivity error control was relaxed on a retry.
    pub statistics: SolverStatistics,
}

impl SolveOutcome {
    /// Assemble the outcome from what an engine produced.
    ///
    /// The one place a solve's payloads are laid out, so the four payload
    /// combinations cannot drift apart in the fields they carry.
    fn from_parts(
        trajectory: DenseTrajectory,
        y_event: Option<Vec<f64>>,
        sensitivities: Option<Vec<Vec<f64>>>,
        statistics: SolverStatistics,
    ) -> Self {
        Self {
            n_rows: trajectory.n_rows,
            n_times: trajectory.n_cols(),
            t: trajectory.t,
            y: trajectory.y,
            yp: trajectory.yp,
            sensitivities,
            t_event: trajectory.t_event,
            y_event,
            flag: trajectory.flag,
            statistics,
        }
    }
}

/// What to integrate over and which payloads to report, shared by every input
/// set of a batch.
///
/// The two payload axes are fields rather than one entry point each: `outputs`
/// swaps the trajectory's rows from states to the model's output variables, and
/// `sensitivities` adds the forward-sensitivity blocks. They compose, so a third
/// axis costs a field rather than doubling the entry points.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SolveRequest<'a> {
    /// Times the solution is reported at.
    pub t_eval: &'a [f64],
    /// Discontinuity times the integrator lands on exactly and restarts from.
    /// Every entry must also appear in `t_eval`; those that do not are
    /// integrated through.
    pub t_stop: &'a [f64],
    /// Report the model's output variables instead of the full state.
    pub outputs: bool,
    /// Solve the forward-sensitivity system, seeded per set by
    /// [`InputSet::y0_sens`].
    pub sensitivities: bool,
}

impl<'a> SolveRequest<'a> {
    /// A state-trajectory request over `t_eval`: no stop times, no
    /// sensitivities.
    pub const fn new(t_eval: &'a [f64]) -> Self {
        Self {
            t_eval,
            t_stop: &[],
            outputs: false,
            sensitivities: false,
        }
    }

    /// Land on, and restart from, each of `t_stop`.
    pub const fn with_stop_times(mut self, t_stop: &'a [f64]) -> Self {
        self.t_stop = t_stop;
        self
    }

    /// Report output-variable rows rather than states.
    pub const fn with_outputs(mut self) -> Self {
        self.outputs = true;
        self
    }

    /// Solve the forward-sensitivity system alongside the trajectory.
    pub const fn with_sensitivities(mut self) -> Self {
        self.sensitivities = true;
        self
    }
}

/// One input set of a solve: where the state starts and what the parameters are.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct InputSet<'a> {
    /// Initial state, one entry per state.
    pub y0: &'a [f64],
    /// Flat input-parameter vector.
    pub inputs: &'a [f64],
    /// `dy0/dp` in column-major `n_states x k` order over the requested
    /// sensitivity subset; empty is the all-zero seed. Read only when the
    /// request asks for sensitivities.
    pub y0_sens: &'a [f64],
}

impl<'a> InputSet<'a> {
    /// An input set carrying the all-zero `dy0/dp` seed.
    pub const fn new(y0: &'a [f64], inputs: &'a [f64]) -> Self {
        Self {
            y0,
            inputs,
            y0_sens: &[],
        }
    }

    /// Seed the sensitivity system with `y0_sens` rather than with zeros.
    pub const fn with_sens_seed(mut self, y0_sens: &'a [f64]) -> Self {
        self.y0_sens = y0_sens;
        self
    }
}

/// Prepare-once/execute-many handle for repeated solves of the same model.
///
/// Holds the shared immutable `CompiledModel`, the tolerances and integrator
/// options, pre-converted sparsity patterns, and the system sizes. It is *not*
/// a fully specified problem: `y0`, `t_eval` and the parameter vector all arrive
/// per call, and each call to [`solve`](Self::solve) builds its own
/// `Workspace`, its own diffsol `OdeSolverProblem` and its own diffsol solver
/// from them. Carrying no
/// per-solve mutable state is what makes this `Send + Sync` and safe to share
/// across threads.
pub struct PreparedSolver {
    compiled: Arc<CompiledModel>,
    rtol: f64,
    atol: Vec<f64>,
    /// Multiplier applied to `atol` on differential rows only.
    sens_atol_factor: f64,
    /// Integrator tuning, stamped onto every problem this handle builds.
    options: SolverOptions,
    /// Store `yp` alongside `y` on the state-trajectory paths.
    store_yp: bool,
    /// True where the mass-matrix row is empty or all zeros.
    algebraic_rows: Vec<bool>,
    /// M's diagonal, or `None` if M has an off-diagonal entry.
    mass_diagonal: Option<Vec<f64>>,
    /// Shared with every solve's `Equations`, whose operator views borrow them;
    /// `Arc` because this handle is `Send + Sync` and outlives every solve.
    jac_sparsity: Arc<FaerSparsity>,
    mass_sparsity: Arc<FaerSparsity>,
    /// M's values in `mass_sparsity` order, precomputed once per problem.
    mass_csc_values: Arc<[f64]>,
    context: FaerContext,
    n_states: usize,
    n_params: usize,
    n_events: usize,
    n_event_outputs: usize,
    n_outputs: usize,
    /// `0..n_params`, the identity subset the plain solve path uses.
    all_param_indices: Arc<[usize]>,
    /// The configured sensitivity subset, precomputed so no solve allocates it.
    sens_param_indices: Arc<[usize]>,
}

impl std::fmt::Debug for PreparedSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSolver")
            .field("n_states", &self.n_states)
            .field("n_params", &self.n_params)
            .field("n_events", &self.n_events)
            .field("n_outputs", &self.n_outputs)
            .finish_non_exhaustive()
    }
}

impl PreparedSolver {
    /// Build a prepared problem from a compiled model and tolerances.
    ///
    /// Performs all expensive one-time setup: sparsity conversion and size
    /// extraction. Takes the shared artifact, which is what it keeps -- each
    /// solve mints its own [`Workspace`] -- though an evaluator converts in.
    pub fn new(
        model: impl Into<Arc<CompiledModel>>,
        rtol: f64,
        atol: &[f64],
    ) -> Result<Self, CoreError> {
        let model = model.into();
        let n_states = model.n_states();
        if atol.len() != n_states {
            return Err(CoreError::AtolLength {
                got: atol.len(),
                expected: n_states,
            });
        }
        let n_params = model.n_params();
        let n_events = model.n_events();
        let n_event_outputs = model.total_event_len();
        let n_outputs = model.total_output_len();

        let jac_sparsity = Arc::new(csc_to_faer_sparsity(model.csc_sparsity()));
        let (mass_sparsity, mass_csc_values) = csr_mass_to_faer_csc(model.mass_matrix());
        let mass_sparsity = Arc::new(mass_sparsity);
        let mass_csc_values: Arc<[f64]> = mass_csc_values.into();

        let mass = model.mass_matrix();
        if mass.indptr().len() != n_states + 1 {
            return Err(CoreError::Csr(format!(
                "mass matrix has {} rows but the model has {n_states} states",
                mass.indptr().len().saturating_sub(1)
            )));
        }
        // An all-zero (or empty) mass row is an algebraic state. `all` on an
        // empty slice is true, so both cases fall out of the same test.
        let algebraic_rows: Vec<bool> = (0..n_states)
            .map(|i| {
                mass.data()[mass.indptr()[i]..mass.indptr()[i + 1]]
                    .iter()
                    .all(|v| *v == 0.0)
            })
            .collect();
        let mass_diagonal = mass_diagonal(mass);

        let compiled = model;
        let all_param_indices: Arc<[usize]> = (0..n_params).collect();
        let sens_param_indices: Arc<[usize]> = compiled.sens_param_indices().into();
        let context = FaerContext::default();

        Ok(Self {
            compiled,
            rtol,
            atol: atol.to_vec(),
            sens_atol_factor: DEFAULT_SENS_ATOL_FACTOR,
            options: SolverOptions::default(),
            store_yp: false,
            algebraic_rows,
            mass_diagonal,
            jac_sparsity,
            mass_sparsity,
            mass_csc_values,
            context,
            n_states,
            n_params,
            n_events,
            n_event_outputs,
            n_outputs,
            all_param_indices,
            sens_param_indices,
        })
    }

    /// Build a fresh local `Equations` for one solve.
    ///
    /// The equations share the immutable compiled model and the supplied solve-local
    /// workspace, and carry a clone of the solve's input parameters.
    fn build_eqn(
        &self,
        y0: &[f64],
        y0_sens: &[f64],
        ws: &Rc<RefCell<Workspace>>,
        with_output: bool,
        inputs: &[f64],
        sens_params: &Arc<[usize]>,
    ) -> Equations {
        Equations {
            compiled: Arc::clone(&self.compiled),
            ws: Rc::clone(ws),
            params: inputs.to_vec(),
            sens_params: Arc::clone(sens_params),
            y0: y0.to_vec(),
            y0_sens: y0_sens.to_vec(),
            jac_sparsity: Arc::clone(&self.jac_sparsity),
            mass_sparsity: Arc::clone(&self.mass_sparsity),
            mass_csc_values: Arc::clone(&self.mass_csc_values),
            context: self.context,
            n_states: self.n_states,
            n_event_outputs: self.n_event_outputs,
            n_outputs: self.n_outputs,
            with_output,
        }
    }

    /// Build the diffsol problem for one solve.
    ///
    /// The single place tolerances, `t0`, the parameter vector and
    /// [`SolverOptions`] are stamped onto a problem, so a new knob reaches
    /// every path at once.
    ///
    /// `sens_scales` carries the per-parameter scales when forward
    /// sensitivities are under error control, and is `None` on the plain paths
    /// and on the relaxed retry.
    ///
    /// # Errors
    /// Propagates whichever `build_from_eqn` rejects the equations for.
    fn build_problem(
        &self,
        eqn: Equations,
        t0: f64,
        p: Vec<f64>,
        sens_scales: Option<Vec<f64>>,
    ) -> Result<OdeSolverProblem<Equations>, CoreError> {
        let mut builder = OdeBuilder::<FaerSparseMat<f64>>::new()
            .rtol(self.rtol)
            .atol(self.atol.clone());
        if let Some(scales) = sens_scales {
            builder = builder
                .sens_rtol(self.rtol)
                .sens_atol(self.sens_atol())
                .param_scales(scales);
        }
        Ok(self.options.apply(builder.t0(t0).p(p).build_from_eqn(eqn)?))
    }

    /// Set the multiplier applied to `atol` on differential rows when forming
    /// the forward-sensitivity tolerance floor.
    ///
    /// # Errors
    /// Returns [`CoreError::SensAtolFactor`] if `factor` is not finite and > 0.
    pub fn with_sens_atol_factor(mut self, factor: f64) -> Result<Self, CoreError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CoreError::SensAtolFactor { got: factor });
        }
        self.sens_atol_factor = factor;
        Ok(self)
    }

    /// Set the integrator tuning applied to every problem this template builds.
    ///
    /// # Errors
    /// Returns [`CoreError::SolverOption`] if any option is out of range.
    pub fn with_options(mut self, options: SolverOptions) -> Result<Self, CoreError> {
        options.validate()?;
        self.options = options;
        Ok(self)
    }

    /// Store the state time derivatives (`yp`) alongside `y` on the
    /// state-trajectory paths, giving downstream cubic-Hermite interpolation
    /// its knot slopes. Doubles trajectory memory, so it is opt-in; the
    /// output-variable paths never store it (they carry no state trajectory),
    /// and nor do grids past [`MAX_HERMITE_COLUMNS`].
    #[must_use]
    pub const fn with_store_yp(mut self, store_yp: bool) -> Self {
        self.store_yp = store_yp;
        self
    }

    /// Whether a solve reporting `n_columns` columns stores `yp` alongside `y`.
    ///
    /// Opting in asks for the slopes where they buy accuracy, not for them
    /// unconditionally; see [`MAX_HERMITE_COLUMNS`].
    const fn store_yp(&self, n_columns: usize) -> bool {
        self.store_yp && n_columns <= MAX_HERMITE_COLUMNS
    }

    /// Mask of algebraic states, one entry per state.
    fn algebraic_rows(&self) -> &[bool] {
        &self.algebraic_rows
    }

    /// Per-state sensitivity absolute tolerance.
    ///
    /// Differential rows carry the tightened floor; algebraic rows keep the
    /// state `atol`, because tightening them fails DAE sensitivity solves.
    fn sens_atol(&self) -> Vec<f64> {
        self.atol
            .iter()
            .zip(self.algebraic_rows())
            .map(|(a, is_algebraic)| {
                if *is_algebraic {
                    *a
                } else {
                    a * self.sens_atol_factor
                }
            })
            .collect()
    }

    /// Per-parameter scales for the sensitivity absolute tolerances, `atol / |scale|`.
    ///
    /// Uses each parameter's own magnitude, which is the scale `dy/dp_j` is expressed in.
    /// Zero and non-finite inputs fall back to 1.0: diffsol rejects them, and a parameter
    /// that is currently zero carries no magnitude information to scale by.
    ///
    /// `sens_inputs` is subset-space (length `k`, already narrowed to the requested
    /// sensitivity columns), not the global `n_params`-length input vector.
    fn param_scales(sens_inputs: &[f64]) -> Vec<f64> {
        sens_inputs
            .iter()
            .map(|v| {
                if v.is_finite() && *v != 0.0 {
                    v.abs()
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// BDF dense-solve kernel behind [`solve`](Self::solve).
    ///
    /// Builds a fresh `Workspace`, runs the timed IC / solver-setup sequence,
    /// then steps each `t_stop` segment in turn, interpolating every output
    /// column straight into the trajectory. diffsol's `solve_dense` would
    /// return the same columns via an intermediate zero-initialised dense
    /// matrix; at DFN size that allocation and its copy-out are a measurable
    /// slice of the whole solve, so the loop writes the final layout directly.
    fn run_dense(
        &self,
        times: TimePlan<'_>,
        y0: &[f64],
        inputs: &[f64],
    ) -> Result<(DenseTrajectory, Option<Vec<f64>>, SolverStatistics), CoreError> {
        let started = std::time::Instant::now();
        let ws = Rc::new(RefCell::new(self.compiled.create_workspace()));
        let eqn = self.build_eqn(y0, &[], &ws, false, inputs, &self.all_param_indices);

        let problem = self.build_problem(eqn, times.eval[0], inputs.to_vec(), None)?;

        let ic_start = std::time::Instant::now();
        let state = problem.bdf_state::<ReusedFaerLu>()?;
        let ic_time_secs = ic_start.elapsed().as_secs_f64();
        let setup_start = std::time::Instant::now();
        let mut solver = problem.bdf_solver::<ReusedFaerLu>(state)?;
        let solver_setup_time_secs = setup_start.elapsed().as_secs_f64();

        let store_yp = self.store_yp(times.eval.len());
        let mut trajectory =
            DenseTrajectory::with_capacity(self.n_states, times.eval.len(), store_yp);
        let mut interp = self.context.vector_zeros::<FaerVec<f64>>(self.n_states);
        let mut interp_dy =
            store_yp.then(|| self.context.vector_zeros::<FaerVec<f64>>(self.n_states));
        let mut restarter = BreakpointRestarter::new(self);
        let root = drive_segments(
            &mut solver,
            times,
            |solver, t_next| restarter.restart(solver, t_next),
            |solver, column| {
                match column {
                    ColumnSource::Interpolated(t) => {
                        solver.interpolate_inplace(t, &mut interp)?;
                        if let Some(dy) = interp_dy.as_mut() {
                            solver.interpolate_dy_inplace(t, dy)?;
                        }
                        trajectory.push_column(
                            t,
                            interp.as_slice(),
                            interp_dy.as_ref().map(VectorHost::as_slice),
                        );
                    },
                    ColumnSource::CurrentState(t) => {
                        // interpolate_dy_inplace rejects t >= state.t; state.dy
                        // is already wound back to an event root alongside y.
                        let state = solver.state();
                        trajectory.push_column(
                            t,
                            state.y.as_slice(),
                            store_yp.then(|| state.dy.as_slice()),
                        );
                    },
                }
                Ok(())
            },
        )?;

        let y_root = root.map(|t_root| {
            trajectory.t_event = Some(t_root);
            trajectory.flag = 1;
            solver.state().y.as_slice().to_vec()
        });

        let mut statistics = SolverStatistics::from(solver.get_statistics());
        statistics.ic_time_secs = ic_time_secs;
        statistics.solver_setup_time_secs = solver_setup_time_secs;
        statistics.integration_time_secs = started.elapsed().as_secs_f64();

        Ok((trajectory, y_root, statistics))
    }

    /// Validate caller-supplied solve arguments against the model dimensions.
    ///
    /// Returns a [`CoreError`] (surfaced as `ValueError` at the Python boundary)
    /// for an empty or non-increasing `t_eval`, a mismatched initial state, or a
    /// mismatched packed input array, so malformed arguments produce ordinary
    /// errors instead of panicking deep inside integration.
    ///
    /// The ordering check is the one diffsol's `solve_dense` used to run for us;
    /// [`drive_segments`] drains the grid in order and would otherwise report a
    /// truncated trajectory as a successful solve.
    const fn validate_args(
        &self,
        t_eval: &[f64],
        y0: &[f64],
        inputs: &[f64],
    ) -> Result<(), CoreError> {
        if t_eval.is_empty() {
            return Err(CoreError::EmptyTimePoints);
        }
        // Indexed rather than `windows(2)` to stay const; equal times are the
        // discontinuity brackets, so only a decrease (or a NaN) is an error.
        let mut i = 1;
        while i < t_eval.len() {
            let (previous, got) = (t_eval[i - 1], t_eval[i]);
            if got < previous || got.is_nan() || previous.is_nan() {
                return Err(CoreError::UnsortedTimePoints {
                    index: i,
                    got,
                    previous,
                });
            }
            i += 1;
        }
        if y0.len() != self.n_states {
            return Err(CoreError::Y0Length {
                got: y0.len(),
                expected: self.n_states,
            });
        }
        if inputs.len() != self.n_params {
            return Err(CoreError::InputsLength {
                got: inputs.len(),
                expected: self.n_params,
            });
        }
        Ok(())
    }

    /// Reject an output-variable solve on a model that registered none.
    ///
    /// The trajectory width comes from `n_outputs`, while the operator the
    /// kernels branch on is only minted when there is an output to evaluate; a
    /// zero-output model would otherwise report state values under a zero row
    /// count.
    const fn validate_has_outputs(&self) -> Result<(), CoreError> {
        if self.n_outputs == 0 {
            return Err(CoreError::NoOutputVariables);
        }
        Ok(())
    }

    /// Integrate the model, returning whichever payloads `request` asked for.
    ///
    /// The one solve entry point: the request's payload flags pick the engine,
    /// and every combination of them lands in one [`SolveOutcome`]. Each call
    /// constructs its own `Workspace` and its own BDF solver, which is what
    /// makes concurrent calls through a shared `&self` sound.
    pub fn solve(
        &self,
        request: SolveRequest<'_>,
        set: InputSet<'_>,
    ) -> Result<SolveOutcome, CoreError> {
        let InputSet {
            y0,
            inputs,
            y0_sens,
        } = set;
        self.validate_args(request.t_eval, y0, inputs)?;
        if request.sensitivities {
            self.validate_y0_sens(y0_sens)?;
        }
        if request.outputs {
            self.validate_has_outputs()?;
        }
        let times = TimePlan::new(request.t_eval, request.t_stop);

        if request.sensitivities {
            let (trajectory, sensitivities, y_event, statistics) =
                self.run_dense_sensitivities(times, y0, y0_sens, inputs, request.outputs)?;
            self.debug_assert_rows(&trajectory, request.outputs);
            return Ok(SolveOutcome::from_parts(
                trajectory,
                y_event,
                Some(sensitivities),
                statistics,
            ));
        }

        let (trajectory, y_event, statistics) = if request.outputs {
            self.run_dense_outputs(times, y0, inputs)?
        } else {
            self.run_dense(times, y0, inputs)?
        };
        self.debug_assert_rows(&trajectory, request.outputs);
        Ok(SolveOutcome::from_parts(
            trajectory, y_event, None, statistics,
        ))
    }

    /// Check a trajectory's rows against what the request asked for.
    ///
    /// One layout contract covering every payload combination, where the
    /// output-variable path used to assert its own and the others none.
    fn debug_assert_rows(&self, trajectory: &DenseTrajectory, outputs: bool) {
        let expected = if outputs {
            self.n_outputs
        } else {
            self.n_states
        };
        debug_assert_eq!(
            trajectory.n_rows, expected,
            "trajectory row count mismatch: expected {expected}, got {}",
            trajectory.n_rows,
        );
    }

    /// Run the dense output-variable solve.
    ///
    /// Integrates plain states and batch-evaluates the output tapes over staged
    /// windows of [`OUTPUT_BATCH_LANES`] points, amortising interpreter
    /// dispatch; values match the per-point path bitwise. The returned state is
    /// always the full state where the trajectory ends (root time on an event,
    /// stop time otherwise) — the only full state an outputs-only caller can
    /// restart from.
    fn run_dense_outputs(
        &self,
        times: TimePlan<'_>,
        y0: &[f64],
        inputs: &[f64],
    ) -> Result<(DenseTrajectory, Option<Vec<f64>>, SolverStatistics), CoreError> {
        let started = std::time::Instant::now();
        let ws = Rc::new(RefCell::new(self.compiled.create_workspace()));
        // with_output stays false: diffsol integrates states, outputs are
        // evaluated in windows below.
        let eqn = self.build_eqn(y0, &[], &ws, false, inputs, &self.all_param_indices);

        let problem = self.build_problem(eqn, times.eval[0], inputs.to_vec(), None)?;

        let ic_start = std::time::Instant::now();
        let state = problem.bdf_state::<ReusedFaerLu>()?;
        let ic_time_secs = ic_start.elapsed().as_secs_f64();
        let setup_start = std::time::Instant::now();
        let mut solver = problem.bdf_solver::<ReusedFaerLu>(state)?;
        let solver_setup_time_secs = setup_start.elapsed().as_secs_f64();

        let n_states = self.n_states;
        let n_out_total = self.n_outputs;
        let mut window = OutputBatchWindow::new(n_states, n_out_total, times.eval.len());
        let mut interp = self.context.vector_zeros::<FaerVec<f64>>(n_states);
        let mut restarter = BreakpointRestarter::new(self);
        let root = drive_segments(
            &mut solver,
            times,
            |solver, t_next| restarter.restart(solver, t_next),
            |solver, column| {
                match column {
                    ColumnSource::Interpolated(t) => {
                        solver.interpolate_inplace(t, &mut interp)?;
                        window.stage(t, interp.as_slice());
                    },
                    ColumnSource::CurrentState(t) => {
                        window.stage(t, solver.state().y.as_slice());
                    },
                }
                window.flush_if_full(&self.compiled, &mut ws.borrow_mut(), inputs);
                Ok(())
            },
        )?;
        window.flush(&self.compiled, &mut ws.borrow_mut(), inputs);

        let (t_event, flag) = root.map_or((None, 0), |t_root| (Some(t_root), 1));
        // Always the full state, the only one an outputs-only caller can
        // restart from; on a root the state is already wound back to it.
        let y_event = Some(solver.state().y.as_slice().to_vec());

        let mut statistics = SolverStatistics::from(solver.get_statistics());
        statistics.ic_time_secs = ic_time_secs;
        statistics.solver_setup_time_secs = solver_setup_time_secs;
        statistics.integration_time_secs = started.elapsed().as_secs_f64();

        let (t, y) = window.into_trajectory();
        Ok((
            DenseTrajectory {
                t,
                y,
                yp: None,
                n_rows: n_out_total,
                t_event,
                flag,
            },
            y_event,
            statistics,
        ))
    }

    /// Run the dense sensitivity solve, retrying once with sensitivities
    /// excluded from error control if the controlled solve fails.
    ///
    /// Stiff DAEs can fail under a tightened sensitivity floor; the retry keeps
    /// this path no worse than excluding sensitivities from error control
    /// entirely, and flags the downgrade in the returned statistics.
    #[allow(clippy::type_complexity)]
    fn run_dense_sensitivities(
        &self,
        times: TimePlan<'_>,
        y0: &[f64],
        y0_sens: &[f64],
        inputs: &[f64],
        with_output: bool,
    ) -> Result<
        (
            DenseTrajectory,
            Vec<Vec<f64>>,
            Option<Vec<f64>>,
            SolverStatistics,
        ),
        CoreError,
    > {
        // Spans both attempts, as a caller-side timer around this call would.
        let started = std::time::Instant::now();
        let mut result =
            match self.run_dense_sensitivities_inner(times, y0, y0_sens, inputs, with_output, true)
            {
                Ok(result) => result,
                // Only an integration failure is worth retrying; a config error
                // fails the relaxed attempt identically and must surface as-is.
                Err(controlled) if !matches!(controlled, CoreError::Diffsol(_)) => {
                    return Err(controlled);
                },
                Err(controlled) => {
                    match self.run_dense_sensitivities_inner(
                        times,
                        y0,
                        y0_sens,
                        inputs,
                        with_output,
                        false,
                    ) {
                        Ok((trajectory, sens_flat, y_root, mut statistics)) => {
                            statistics.sens_error_control_relaxed = true;
                            (trajectory, sens_flat, y_root, statistics)
                        },
                        // Both attempts failed: the controlled cause is the useful one.
                        Err(relaxed) => {
                            return Err(CoreError::SensRetryFailed {
                                controlled: controlled.to_string(),
                                relaxed: relaxed.to_string(),
                            });
                        },
                    }
                },
            };
        result.3.integration_time_secs = started.elapsed().as_secs_f64();
        Ok(result)
    }

    /// BDF dense sensitivity-solve kernel behind
    /// [`SolveRequest::sensitivities`], reporting output-variable rows rather
    /// than states when `with_output` is set.
    ///
    /// Mirrors [`run_dense`](Self::run_dense) on the diffsol forward-sensitivity
    /// chain (`bdf_state_sens` / `bdf_solver_sens`), interpolating each output
    /// column and its `k` sensitivity columns straight into the flat trajectory.
    /// Returns the trajectory, one flat block per parameter, the full state at
    /// the root time (if an event fired), and timing-enriched statistics.
    #[allow(clippy::type_complexity)]
    fn run_dense_sensitivities_inner(
        &self,
        times: TimePlan<'_>,
        y0: &[f64],
        y0_sens: &[f64],
        inputs: &[f64],
        with_output: bool,
        sens_error_control: bool,
    ) -> Result<
        (
            DenseTrajectory,
            Vec<Vec<f64>>,
            Option<Vec<f64>>,
            SolverStatistics,
        ),
        CoreError,
    > {
        let sens_params = &self.sens_param_indices;
        if sens_params.is_empty() {
            return Err(CoreError::NoSensitivityParams);
        }
        let ws = Rc::new(RefCell::new(self.compiled.create_workspace()));
        let eqn = self.build_eqn(y0, y0_sens, &ws, with_output, inputs, sens_params);
        // The ops report the subset width, so p and the scales narrow with them.
        let sens_inputs: Vec<f64> = sens_params.iter().map(|&i| inputs[i]).collect();

        // param_scales is IDAS's pbar; without it Chen2020 collapses to ~97k steps.
        // sens_atol tightens differential rows only, per the tolerance-structure spec.
        let scales = sens_error_control.then(|| Self::param_scales(&sens_inputs));
        let problem = self.build_problem(eqn, times.eval[0], sens_inputs, scales)?;

        let ic_start = std::time::Instant::now();
        let state = problem.bdf_state_sens::<ReusedFaerLu>()?;
        let ic_time_secs = ic_start.elapsed().as_secs_f64();
        let setup_start = std::time::Instant::now();
        let mut solver = problem.bdf_solver_sens::<ReusedFaerLu>(state)?;
        let solver_setup_time_secs = setup_start.elapsed().as_secs_f64();

        let n_columns = sens_params.len();
        debug_assert_eq!(
            n_columns,
            self.compiled.n_sens_params(),
            "sens subset width diverged from the compiled model",
        );
        let n_rows = if with_output {
            self.n_outputs
        } else {
            self.n_states
        };
        let n_points = times.eval.len();
        // Output mode never stores yp: its trajectory holds outputs, not states.
        let store_yp = self.store_yp(n_points) && !with_output;
        let mut trajectory = DenseTrajectory::with_capacity(n_rows, n_points, store_yp);
        let mut sens_flat: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rows * n_points); n_columns];
        let mut y_column = self.context.vector_zeros::<FaerVec<f64>>(self.n_states);
        let mut dy_column =
            store_yp.then(|| self.context.vector_zeros::<FaerVec<f64>>(self.n_states));
        let mut sens_columns: Vec<FaerVec<f64>> = (0..n_columns)
            .map(|_| self.context.vector_zeros::<FaerVec<f64>>(self.n_states))
            .collect();
        let mut out_values = self.context.vector_zeros::<FaerVec<f64>>(self.n_outputs);
        let mut out_chain = self.context.vector_zeros::<FaerVec<f64>>(self.n_outputs);
        let mut out_direct = self.context.vector_zeros::<FaerVec<f64>>(self.n_outputs);
        let mut unit_param = self.context.vector_zeros::<FaerVec<f64>>(n_columns);

        let mut restarter = BreakpointRestarter::new(self);
        let root = drive_segments(
            &mut solver,
            times,
            |solver, t_next| restarter.restart_sens(solver, t_next),
            |solver, column| {
                let t = column.time();
                match column {
                    ColumnSource::Interpolated(_) => {
                        solver.interpolate_inplace(t, &mut y_column)?;
                        solver.interpolate_sens_inplace(t, &mut sens_columns)?;
                        if let Some(dy) = dy_column.as_mut() {
                            solver.interpolate_dy_inplace(t, dy)?;
                        }
                    },
                    ColumnSource::CurrentState(_) => {
                        let state = solver.state();
                        debug_assert_eq!(
                            state.s.len(),
                            sens_columns.len(),
                            "augmented state carries a different sens width"
                        );
                        y_column.copy_from(state.y);
                        if let Some(dy) = dy_column.as_mut() {
                            dy.copy_from(state.dy);
                        }
                        for (dst, src) in sens_columns.iter_mut().zip(state.s) {
                            dst.copy_from(src);
                        }
                    },
                }
                // Output mode reports g and dg/dp + dg/dy · y_s, the chain rule
                // diffsol's own dense sensitivity write-out applies.
                if let Some(out) = solver.problem().eqn.out() {
                    out.call_inplace(&y_column, t, &mut out_values);
                    trajectory.push_column(t, out_values.as_slice(), None);
                    for (j, (flat, s_j)) in sens_flat.iter_mut().zip(&sens_columns).enumerate() {
                        out.jac_mul_inplace(&y_column, t, s_j, &mut out_chain);
                        unit_param.set_index(j, 1.0);
                        out.sens_mul_inplace(&y_column, t, &unit_param, &mut out_direct);
                        unit_param.set_index(j, 0.0);
                        flat.extend(
                            out_chain
                                .as_slice()
                                .iter()
                                .zip(out_direct.as_slice())
                                .map(|(chain, direct)| chain + direct),
                        );
                    }
                } else {
                    trajectory.push_column(
                        t,
                        y_column.as_slice(),
                        dy_column.as_ref().map(VectorHost::as_slice),
                    );
                    for (flat, s_j) in sens_flat.iter_mut().zip(&sens_columns) {
                        flat.extend_from_slice(s_j.as_slice());
                    }
                }
                Ok(())
            },
        )?;

        let mut y_root = root.map(|t_root| {
            trajectory.t_event = Some(t_root);
            trajectory.flag = 1;
            solver.state().y.as_slice().to_vec()
        });
        // In output mode the trajectory never holds states, so the terminal
        // state is the caller's only one.
        if with_output && y_root.is_none() {
            y_root = Some(solver.state().y.as_slice().to_vec());
        }
        let mut statistics = SolverStatistics::from(solver.get_statistics());
        statistics.ic_time_secs = ic_time_secs;
        statistics.solver_setup_time_secs = solver_setup_time_secs;

        Ok((trajectory, sens_flat, y_root, statistics))
    }

    /// Validate the caller-supplied `dy0/dp` seed.
    ///
    /// Empty is the "no parameter reaches an initial condition" case; anything
    /// else must be a column-major `n_states x k` block over the requested
    /// sensitivity subset.
    ///
    /// A model compiled without sensitivity parameters is reported as such
    /// first: its expected width is 0, so the length complaint would otherwise
    /// mask the real configuration mistake behind "must be empty or 0".
    fn validate_y0_sens(&self, y0_sens: &[f64]) -> Result<(), CoreError> {
        if self.sens_param_indices.is_empty() {
            return Err(CoreError::NoSensitivityParams);
        }
        let expected = self.n_states * self.sens_param_indices.len();
        if y0_sens.is_empty() || y0_sens.len() == expected {
            return Ok(());
        }
        Err(CoreError::Y0SensLength {
            got: y0_sens.len(),
            expected,
        })
    }

    /// Flatten a dense matrix `[nrows × ncols]` in column-major order:
    /// `out[col * nrows + row]` (time-outer/row-inner).
    ///
    /// Only the solve paths that still call diffsol's `solve_dense` need this,
    /// which is now just the LU-equivalence test.
    #[cfg(test)]
    fn dense_to_column_major(
        m: &<FaerVec<f64> as diffsol::vector::DefaultDenseMatrix>::M,
    ) -> Vec<f64> {
        use diffsol::MatrixCommon;

        let mut flat = Vec::with_capacity(m.nrows() * m.ncols());
        for j in 0..m.ncols() {
            flat.extend_from_slice(m.inner().col_as_slice(j));
        }
        flat
    }
}

/// How many interpolation points are staged before one batched output
/// evaluation. Bounds the extra state storage at `n_states * OUTPUT_BATCH_LANES`
/// while still amortising per-instruction dispatch across the window.
const OUTPUT_BATCH_LANES: usize = 128;

/// Staging buffer for the batched output-variable path.
///
/// Collects interpolated `(t, y)` columns until [`OUTPUT_BATCH_LANES`] are
/// pending, then evaluates every output tape across the window in one
/// `eval_batch` pass and appends the results to the growing trajectory.
struct OutputBatchWindow {
    n_states: usize,
    n_out_total: usize,
    staged_t: Vec<f64>,
    /// `(n_states, k)` F-contiguous staged states.
    staged_y: Vec<f64>,
    /// `(n_out_total, k)` F-contiguous per-window results.
    window_out: Vec<f64>,
    t_out: Vec<f64>,
    outputs: Vec<f64>,
}

impl OutputBatchWindow {
    fn new(n_states: usize, n_out_total: usize, n_points_hint: usize) -> Self {
        Self {
            n_states,
            n_out_total,
            staged_t: Vec::with_capacity(OUTPUT_BATCH_LANES),
            staged_y: Vec::with_capacity(n_states * OUTPUT_BATCH_LANES),
            window_out: vec![0.0; n_out_total * OUTPUT_BATCH_LANES],
            t_out: Vec::with_capacity(n_points_hint),
            outputs: Vec::with_capacity(n_out_total * n_points_hint),
        }
    }

    fn stage(&mut self, t: f64, y: &[f64]) {
        debug_assert!(self.staged_t.len() < OUTPUT_BATCH_LANES);
        self.staged_t.push(t);
        self.staged_y.extend_from_slice(&y[..self.n_states]);
    }

    fn flush_if_full(&mut self, compiled: &CompiledModel, ws: &mut Workspace, inputs: &[f64]) {
        if self.staged_t.len() == OUTPUT_BATCH_LANES {
            self.flush(compiled, ws, inputs);
        }
    }

    fn flush(&mut self, compiled: &CompiledModel, ws: &mut Workspace, inputs: &[f64]) {
        let k = self.staged_t.len();
        if k == 0 {
            return;
        }
        compiled.eval_outputs_batch(
            ws,
            k,
            &self.staged_t,
            &self.staged_y,
            inputs,
            &mut self.window_out[..self.n_out_total * k],
        );
        self.outputs
            .extend_from_slice(&self.window_out[..self.n_out_total * k]);
        self.t_out.extend_from_slice(&self.staged_t);
        self.staged_t.clear();
        self.staged_y.clear();
    }

    /// The accumulated `(t, outputs)` trajectory; every window must be flushed.
    fn into_trajectory(self) -> (Vec<f64>, Vec<f64>) {
        debug_assert!(self.staged_t.is_empty(), "unflushed output window");
        (self.t_out, self.outputs)
    }
}

/// Prepare and run a one-shot solve.
///
/// Setup is discarded afterwards, so callers that solve the same model more than
/// once should keep a [`PreparedSolver`] instead. `atol` is per state.
pub fn solve(
    model: impl Into<Arc<CompiledModel>>,
    t_eval: &[f64],
    t_stop: &[f64],
    y0: &[f64],
    inputs: &[f64],
    rtol: f64,
    atol: &[f64],
) -> Result<SolveOutcome, CoreError> {
    let prepared = PreparedSolver::new(model, rtol, atol)?;
    prepared.solve(
        SolveRequest::new(t_eval).with_stop_times(t_stop),
        InputSet::new(y0, inputs),
    )
}

/// M's diagonal, or `None` if any stored nonzero sits off it.
///
/// Only a diagonal M makes `M dy = f` an element-wise divide in
/// [`BreakpointRestarter`]; `PyBaMM`'s discretisation always produces one.
fn mass_diagonal(mass: &CsrData) -> Option<Vec<f64>> {
    let indptr = mass.indptr();
    let mut diagonal = vec![0.0; indptr.len().saturating_sub(1)];
    for (row, entry) in diagonal.iter_mut().enumerate() {
        for k in indptr[row]..indptr[row + 1] {
            if mass.indices()[k] == row {
                *entry = mass.data()[k];
            } else if mass.data()[k] != 0.0 {
                return None;
            }
        }
    }
    Some(diagonal)
}

/// The reusable half of a breakpoint restart.
///
/// One of these lives for a whole solve. Its Newton solvers each keep a
/// [`ReusedFaerLu`], whose entire purpose is to hold a symbolic factorisation
/// across refreshes; building them per segment threw that away, and `PyBaMM`
/// hands down one stop time per output point, so a dense grid restarts
/// hundreds of times. The two solvers stay separate because they factorise
/// different sparsity patterns and sharing one would thrash both.
struct BreakpointRestarter {
    root_solver: NewtonNonlinearSolver<FaerSparseMat<f64>, ReusedFaerLu, NoLineSearch>,
    sens_solver: NewtonNonlinearSolver<FaerSparseMat<f64>, ReusedFaerLu, NoLineSearch>,
    /// M's diagonal, only when the model has no algebraic state and M is
    /// diagonal; `None` leaves `dy` to `set_consistent`.
    ode_mass_diagonal: Option<Vec<f64>>,
    /// Holds `f(t, y)` while `dy` is rebuilt from it.
    scratch: FaerVec<f64>,
}

impl BreakpointRestarter {
    fn new(problem: &PreparedSolver) -> Self {
        let has_algebraic = problem
            .algebraic_rows
            .iter()
            .any(|is_algebraic| *is_algebraic);
        Self {
            root_solver: NewtonNonlinearSolver::new(ReusedFaerLu::default(), NoLineSearch),
            sens_solver: NewtonNonlinearSolver::new(ReusedFaerLu::default(), NoLineSearch),
            ode_mass_diagonal: (!has_algebraic)
                .then(|| problem.mass_diagonal.clone())
                .flatten(),
            scratch: problem
                .context
                .vector_zeros::<FaerVec<f64>>(problem.n_states),
        }
    }

    /// Rebuild `dy` from `M dy = f(t, y)` when nothing else will.
    ///
    /// diffsol's `set_consistent` returns early once it finds no zero diagonal
    /// in M, leaving `dy` holding the slope from *before* the discontinuity.
    /// The next step seeds the BDF difference array straight from it
    /// (`initialise_diff_to_first_order`), so without this the restart
    /// re-derives the old branch — the exact failure the restart exists to
    /// prevent, fixed for DAEs and silently broken for ODEs.
    ///
    /// `t_next` is the incoming segment's first output time. `PyBaMM` brackets
    /// a discontinuity with a pair of times one ULP apart and the restart sits
    /// on the earlier one, so `f(state.t)` is still the old branch; when
    /// `t_next` is within round-off it is the far side of the corner, the
    /// branch the segment actually integrates.
    fn refresh_ode_dy<Eqn>(
        &mut self,
        problem: &OdeSolverProblem<Eqn>,
        state: &mut StateRefMut<'_, FaerVec<f64>>,
        t_next: f64,
    ) where
        Eqn: OdeEquationsImplicit<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>,
    {
        let Some(diagonal) = self.ode_mass_diagonal.as_deref() else {
            return;
        };
        let t = if stop_already_reached(*state.t, t_next) {
            t_next
        } else {
            *state.t
        };
        problem
            .eqn
            .rhs()
            .call_inplace(state.y, t, &mut self.scratch);
        for (i, m) in diagonal.iter().enumerate() {
            state.dy[i] = self.scratch[i] / m;
        }
    }

    /// Restart the integrator at a breakpoint, as IDAKLU's `HandleBreakpoint` does.
    ///
    /// Recomputing the consistent state (the diffsol equivalent of `IDACalcIC`
    /// with `IDA_YA_YDP_INIT`) and a fresh step size is what makes the restart
    /// land on the new branch of the solution.
    fn restart<'a, Eqn, S>(&mut self, solver: &mut S, t_next: f64) -> Result<(), CoreError>
    where
        Eqn: OdeEquationsImplicit<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>
            + 'a,
        S: OdeSolverMethod<'a, Eqn>,
    {
        // Taken before the mutable borrow; the problem outlives the solver.
        let problem = solver.problem();
        let mut state = solver.state_mut();
        state.set_consistent(problem, &mut self.root_solver)?;
        self.refresh_ode_dy(problem, &mut state, t_next);
        state.set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);
        Ok(())
    }

    /// [`Self::restart`] for a solve carrying forward sensitivities.
    ///
    /// The sensitivity difference arrays are seeded from `ds`, which
    /// `set_consistent_augmented` rebuilds from `y` and `dy`; running it after
    /// the `dy` refresh is what keeps the two consistent at the corner.
    fn restart_sens<'a, Eqn, AugEqn, S>(
        &mut self,
        solver: &mut S,
        t_next: f64,
    ) -> Result<(), CoreError>
    where
        Eqn: OdeEquationsImplicit<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>
            + 'a,
        AugEqn: AugmentedOdeEquationsImplicit<Eqn> + std::fmt::Debug,
        S: AugmentedOdeSolverMethod<'a, Eqn, AugEqn>,
    {
        let problem = solver.problem();
        {
            let mut state = solver.state_mut();
            state.set_consistent(problem, &mut self.root_solver)?;
            self.refresh_ode_dy(problem, &mut state, t_next);
        }

        // Reads the y and dy just made consistent, so it has to follow them.
        if let Some((mut state, augmented_eqn)) = solver.state_and_augmented_eqn_mut() {
            state.set_consistent_augmented(problem, augmented_eqn, &mut self.sens_solver)?;
        }

        solver
            .state_mut()
            .set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);
        Ok(())
    }
}

/// The times a solve reports at, together with the stop times inside them.
///
/// `stop` is a subset of `eval`: a stop time absent from the output grid cannot
/// end a segment, because the segment's last output time is what the integrator
/// is told to stop on.
#[derive(Clone, Copy, Debug)]
struct TimePlan<'a> {
    eval: &'a [f64],
    stop: &'a [f64],
}

impl<'a> TimePlan<'a> {
    const fn new(eval: &'a [f64], stop: &'a [f64]) -> Self {
        Self { eval, stop }
    }

    fn segments(&self) -> Vec<Range<usize>> {
        segment_ranges(self.eval, self.stop)
    }
}

/// Where one output column's values come from.
///
/// Columns inside a step are interpolated. The two that land where the
/// integrator already sits are read off the state instead: the final column of
/// a segment, which diffsol may report as reached from either side of its
/// round-off window, and an event's column after the state is wound back to it.
#[derive(Clone, Copy, Debug)]
enum ColumnSource {
    Interpolated(f64),
    CurrentState(f64),
}

impl ColumnSource {
    const fn time(self) -> f64 {
        match self {
            Self::Interpolated(t) | Self::CurrentState(t) => t,
        }
    }
}

/// Step one `t_stop` segment at a time, handing every output column to `emit`.
///
/// Sole home of the stop-time round-off rule and the event semantics, so all
/// three solve lanes report the same times. Returns the root time if an event
/// ended the solve, with the solver wound back to it.
///
/// diffsol's `solve_dense` would route the same columns through a
/// zero-initialised `n_states x n_times` matrix and copy it out again.
fn drive_segments<'a, Eqn, S>(
    solver: &mut S,
    times: TimePlan<'_>,
    mut restart: impl FnMut(&mut S, f64) -> Result<(), CoreError>,
    mut emit: impl FnMut(&S, ColumnSource) -> Result<(), CoreError>,
) -> Result<Option<f64>, CoreError>
where
    Eqn: OdeEquationsImplicit<T = f64, V = FaerVec<f64>, M = FaerSparseMat<f64>, C = FaerContext>
        + 'a,
    S: OdeSolverMethod<'a, Eqn>,
{
    debug_assert!(
        solver.problem().eqn.reset().is_none(),
        "a reset operator would need the apply_reset branch solve_dense has",
    );
    let t_eval = times.eval;
    let mut col = 0;
    for (i, range) in times.segments().into_iter().enumerate() {
        let last = range.end - 1;
        if i > 0 {
            // The incoming segment's first output time: the far side of a
            // ULP-bracketed corner, where the new branch is evaluable.
            restart(solver, t_eval[range.start])?;
        }
        solver.set_stop_time(t_eval[last])?;
        loop {
            let reason = solver.step()?;
            let drain_until = match reason {
                OdeSolverStopReason::InternalTimestep | OdeSolverStopReason::TstopReached => {
                    solver.state().t
                },
                OdeSolverStopReason::RootFound(t_root, _) => t_root,
            };
            while col <= last && t_eval[col] <= drain_until {
                emit(solver, ColumnSource::Interpolated(t_eval[col]))?;
                col += 1;
            }
            match reason {
                OdeSolverStopReason::InternalTimestep => {},
                OdeSolverStopReason::TstopReached => {
                    // diffsol leaves `state.t` a round-off short of tstop, so
                    // the next segment would extrapolate this column.
                    if col == last {
                        emit(solver, ColumnSource::CurrentState(t_eval[last]))?;
                        col += 1;
                    }
                    debug_assert_eq!(col, range.end, "segment ended with undrained output points");
                    break;
                },
                OdeSolverStopReason::RootFound(t_root, _) => {
                    solver.state_mut_back(t_root)?;
                    // A root column only when output points remain, so a root
                    // at the segment's last time adds none.
                    if col <= last {
                        emit(solver, ColumnSource::CurrentState(t_root))?;
                    }
                    return Ok(Some(t_root));
                },
            }
        }
    }
    Ok(None)
}

/// Whether diffsol would already consider a stop at `to` reached from `from`.
///
/// Mirrors the round-off window in its `handle_tstop`, where asking to stop is
/// an error rather than a no-op. `PyBaMM` brackets each constant-time
/// discontinuity with a pair of times one ULP apart, so the pair has to collapse
/// to the earlier one: that is the side the pre-discontinuity branch holds on.
fn stop_already_reached(from: f64, to: f64) -> bool {
    (to - from).abs() <= 100.0 * f64::EPSILON * from.abs().max(to.abs())
}

/// One range of `t_eval` per stretch between consecutive stop times.
///
/// A stop time only ends a segment when it is itself an output time. The Python
/// layer guarantees that by unioning `t_eval` into the output grid, so a stop
/// time that is missing from it means the two disagree, and integrating straight
/// through is safer than shifting an output column off the time it belongs to.
fn segment_ranges(t_eval: &[f64], t_stop: &[f64]) -> Vec<Range<usize>> {
    let last = t_eval.len().saturating_sub(1);
    let mut cuts: Vec<usize> = Vec::new();
    for stop in t_stop {
        let idx = t_eval.partition_point(|t| t < stop);
        // Bit equality, not tolerance: the value was copied from this grid.
        let on_grid = idx < t_eval.len() && t_eval[idx].to_bits() == stop.to_bits();
        if !(on_grid && idx > 0 && idx < last) {
            continue;
        }
        // Keeps cuts increasing, so an unsorted `t_stop` is skipped rather than
        // reversing a range into a slice panic.
        let previous = cuts.last().copied().unwrap_or(0);
        if idx <= previous {
            continue;
        }
        if stop_already_reached(t_eval[previous], t_eval[idx])
            || stop_already_reached(t_eval[idx], t_eval[last])
        {
            continue;
        }
        cuts.push(idx);
    }

    let mut ranges = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for cut in cuts {
        ranges.push(start..cut + 1);
        start = cut + 1;
    }
    ranges.push(start..t_eval.len());
    ranges
}

/// Output columns above which a solve gives up storing `yp`.
///
/// The chord's error falls as the column count squared, so past a few thousand
/// columns it reaches the integration error floor Hermite already sits on while
/// the store's cost keeps growing linearly.
const MAX_HERMITE_COLUMNS: usize = 4096;

/// A column-major trajectory accumulated one output column at a time.
///
/// Segments are solved one at a time so the BDF can restart at each stop time;
/// their columns land here in order, together with the root-time bookkeeping
/// when an event ends the solve.
struct DenseTrajectory {
    t: Vec<f64>,
    y: Vec<f64>,
    /// `dy/dt` columns sharing `y`'s layout; `Some` only when the solve
    /// stores them for downstream Hermite interpolation.
    yp: Option<Vec<f64>>,
    /// Trajectory width: states, or outputs on the output paths.
    n_rows: usize,
    t_event: Option<f64>,
    flag: i32,
}

impl DenseTrajectory {
    fn with_capacity(n_rows: usize, n_points: usize, store_yp: bool) -> Self {
        Self {
            t: Vec::with_capacity(n_points),
            y: Vec::with_capacity(n_rows * n_points),
            yp: store_yp.then(|| Vec::with_capacity(n_rows * n_points)),
            n_rows,
            t_event: None,
            flag: 0,
        }
    }

    fn push_column(&mut self, t: f64, y: &[f64], yp: Option<&[f64]>) {
        debug_assert_eq!(y.len(), self.n_rows, "column width mismatch");
        debug_assert_eq!(
            yp.is_some(),
            self.yp.is_some(),
            "every column must carry yp exactly when the trajectory stores it"
        );
        self.t.push(t);
        self.y.extend_from_slice(y);
        if let (Some(buffer), Some(column)) = (self.yp.as_mut(), yp) {
            debug_assert_eq!(column.len(), self.n_rows, "yp column width mismatch");
            buffer.extend_from_slice(column);
        }
    }

    const fn n_cols(&self) -> usize {
        self.t.len()
    }
}

#[cfg(test)]
mod segment_tests {
    use super::segment_ranges;

    #[test]
    fn no_stop_times_is_one_segment() {
        let t = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[]), vec![0..4]);
    }

    #[test]
    fn an_interior_stop_time_ends_a_segment_on_itself() {
        let t = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[2.0]), vec![0..3, 3..4]);
    }

    #[test]
    fn the_span_endpoints_are_not_interior_stop_times() {
        let t = [0.0, 1.0, 2.0];
        assert_eq!(segment_ranges(&t, &[0.0, 2.0]), vec![0..3]);
    }

    #[test]
    fn a_stop_time_on_the_last_point_leaves_no_trailing_segment() {
        let t = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[2.0, 3.0]), vec![0..3, 3..4]);
    }

    #[test]
    fn stop_times_absent_from_the_output_grid_are_ignored() {
        let t = [0.0, 1.0, 2.0];
        assert_eq!(segment_ranges(&t, &[1.5]), vec![0..3]);
    }

    #[test]
    fn consecutive_output_points_can_each_be_a_stop_time() {
        let t = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[1.0, 2.0]), vec![0..2, 2..3, 3..4]);
    }

    #[test]
    fn a_discontinuity_bracket_collapses_to_its_earlier_side() {
        // PyBaMM brackets `t < 2` with 2.0 and its neighbour one ULP below;
        // diffsol rejects the second as a stop time it has already reached.
        let below = f64::from_bits(2.0f64.to_bits() - 1);
        let t = [0.0, 1.0, below, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[below, 2.0]), vec![0..3, 3..5]);
    }

    #[test]
    fn unsorted_stop_times_are_skipped_not_reversed_into_a_panic() {
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(segment_ranges(&t, &[3.0, 1.0]), vec![0..4, 4..5]);
    }

    #[test]
    fn a_repeated_stop_time_does_not_cut_twice() {
        let t = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(segment_ranges(&t, &[2.0, 2.0]), vec![0..3, 3..4]);
    }

    #[test]
    fn a_stop_time_a_hair_before_the_end_leaves_no_degenerate_segment() {
        let below = f64::from_bits(3.0f64.to_bits() - 1);
        let t = [0.0, 1.0, below, 3.0];
        assert_eq!(segment_ranges(&t, &[below]), vec![0..4]);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use diffsol::ode_solver::sensitivities::SensitivitiesOdeSolverMethod;
    use diffsol::{Matrix, NonLinearOpSens, OdeBuilder, OdeEquations, Op, Vector};

    use super::*;
    use crate::arena::Arena;
    use crate::model::{CompiledModelOptions, ModelEvaluator};
    use crate::node::{CsrData, Node, Shape};

    /// Build a small DAE model with one parameter:
    ///   `dy/dt = -a * y`  (`a = inputs[0]`)
    ///   `0 = y - z`       (algebraic: z follows y)
    /// Mass = diag(1, 0), `n_states` = 2, `n_params` = 1, sens wrt param 0.
    #[cfg(test)]
    pub fn build_small_dae_with_param() -> ModelEvaluator {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let neg_a_y = {
            let neg = arena.alloc(Node::Scalar(-1.0));
            let neg_a = arena.alloc(Node::Mul(neg, a));
            arena.alloc(Node::Mul(neg_a, sv0))
        };
        // Algebraic residual: y - z = 0
        let algebraic_residual = arena.alloc(Node::Sub(sv0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![neg_a_y, algebraic_residual]));

        let mass = CsrData {
            indptr: vec![0, 1, 1], // row 0: (0,0)=1; row 1: empty
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(2, 2),
        };

        ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            2,
            1,
            CompiledModelOptions::new().with_sensitivities(&[0]),
        )
    }

    /// Like `build_small_dae_with_param` but with a second input `b` in the ODE
    /// (`dy/dt = -a*y + b`), with sensitivities requested for `sens`.
    fn build_small_dae_two_inputs(sens: &[usize]) -> ModelEvaluator {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let b = arena.alloc(Node::InputParameter {
            name: "b".to_string(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let ode = {
            let neg = arena.alloc(Node::Scalar(-1.0));
            let neg_a = arena.alloc(Node::Mul(neg, a));
            let neg_a_y = arena.alloc(Node::Mul(neg_a, sv0));
            arena.alloc(Node::Add(neg_a_y, b))
        };
        let algebraic_residual = arena.alloc(Node::Sub(sv0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![ode, algebraic_residual]));

        let mass = CsrData {
            indptr: vec![0, 1, 1], // row 0: (0,0)=1; row 1: empty
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(2, 2),
        };

        ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            2,
            2,
            CompiledModelOptions::new().with_sensitivities(sens),
        )
    }

    /// The two-input fixture with sensitivities requested for `a` (index 0) only.
    fn build_small_dae_two_inputs_one_sens() -> ModelEvaluator {
        build_small_dae_two_inputs(&[0])
    }

    /// Return a `PreparedSolver` ready for a sensitivity solve on the small DAE.
    #[cfg(test)]
    pub fn build_small_dae_prepared_with_sens() -> PreparedSolver {
        let model = build_small_dae_with_param();
        PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed")
    }

    /// The sensitivity blocks of an outcome whose request asked for them.
    #[cfg(test)]
    fn blocks(outcome: &SolveOutcome) -> &[Vec<f64>] {
        outcome
            .sensitivities
            .as_deref()
            .expect("the request asked for sensitivities")
    }

    /// Return `(y0, inputs, t_eval)` for the small-DAE fixture.
    #[cfg(test)]
    pub fn small_dae_setup() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let y0 = vec![1.0, 0.0]; // algebraic IC deliberately inconsistent
        let inputs = vec![1.0]; // a = 1.0
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();
        (y0, inputs, t_eval)
    }

    /// One state, `dy/dt = -a * y`, identity mass. No algebraic row, which is
    /// the branch `set_consistent` returns early from.
    fn build_small_ode_with_param() -> ModelEvaluator {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let rhs = arena.alloc(Node::Mul(neg_a, sv0));

        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(1, 1),
        };

        ModelEvaluator::new_with_options(&arena, rhs, mass, 1, 1, CompiledModelOptions::new())
    }

    #[test]
    fn a_breakpoint_restart_refreshes_dy_on_a_pure_ode() {
        let prepared =
            PreparedSolver::new(build_small_ode_with_param(), 1e-8, &[1e-8]).expect("prepare");
        let inputs = [2.0_f64];
        let y0 = [1.0_f64];

        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        let eqn = prepared.build_eqn(&y0, &[], &ws, false, &inputs, &prepared.all_param_indices);
        let problem = prepared.options.apply(
            OdeBuilder::<FaerSparseMat<f64>>::new()
                .rtol(prepared.rtol)
                .atol(prepared.atol.clone())
                .t0(0.0)
                .p(inputs.to_vec())
                .build_from_eqn(eqn)
                .expect("build"),
        );
        let state = problem.bdf_state::<ReusedFaerLu>().expect("state");
        let mut solver = problem.bdf_solver::<ReusedFaerLu>(state).expect("solver");
        // Step away from t0 so `dy` holds a BDF difference-array slope rather
        // than the seed it was initialised with.
        solver.solve_dense(&[0.0, 0.5, 1.0]).expect("solve");

        let stale = solver.state().dy[0];
        BreakpointRestarter::new(&prepared)
            .restart(&mut solver, 2.0)
            .expect("restart");

        let state = solver.state();
        let expected = -inputs[0] * state.y[0];
        assert!(
            (stale - expected).abs() > 1e-14,
            "fixture is vacuous: dy was already exact before the restart",
        );
        assert!(
            (state.dy[0] - expected).abs() <= 1e-13 * expected.abs(),
            "dy = {} after the restart but f(t, y) = {expected}",
            state.dy[0],
        );
    }

    #[test]
    fn solve_empty_t_eval_returns_error_not_panic() {
        let model = build_small_dae_with_param();
        let (y0, inputs, _t) = small_dae_setup();
        let err = solve(model, &[], &[], &y0, &inputs, 1e-8, &[1e-8, 1e-8])
            .expect_err("empty t_eval must be a returned error, not a panic");
        assert!(
            matches!(err, CoreError::EmptyTimePoints),
            "expected EmptyTimePoints, got: {err:?}"
        );
    }

    #[test]
    fn solve_decreasing_t_eval_returns_error_not_a_truncated_solve() {
        // The segment loop drains the grid in order, so without this check a
        // decrease silently reports the points it did reach as a whole solve.
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        for t_eval in [
            vec![0.0, 2.0, 1.0],
            vec![0.0, 0.5, 0.2],
            vec![0.0, f64::NAN, 1.0],
        ] {
            let err = prepared
                .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
                .expect_err("a decreasing t_eval must be an error, not a short trajectory");
            assert!(
                matches!(err, CoreError::UnsortedTimePoints { .. }),
                "expected UnsortedTimePoints for {t_eval:?}, got: {err:?}"
            );
        }
    }

    #[test]
    fn output_solves_reject_a_model_without_output_variables() {
        // Without outputs the kernels fall through to the state branch, which
        // would report state values under a zero row count.
        let prepared = build_small_dae_prepared_with_sens(); // no add_output
        let (y0, inputs, t_eval) = small_dae_setup();
        let err = prepared
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&y0, &inputs),
            )
            .expect_err("an outputs solve on a model with none must be an error");
        assert!(
            matches!(err, CoreError::NoOutputVariables),
            "expected NoOutputVariables, got: {err:?}"
        );
        let err = prepared
            .solve(
                SolveRequest::new(&t_eval)
                    .with_outputs()
                    .with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect_err("an outputs sensitivity solve on a model with none must be an error");
        assert!(
            matches!(err, CoreError::NoOutputVariables),
            "expected NoOutputVariables, got: {err:?}"
        );
    }

    #[test]
    fn solve_repeated_t_eval_still_solves() {
        // Equal consecutive times bracket a discontinuity; only a decrease is
        // rejected.
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        let t_eval = vec![0.0, 0.2, 0.2, 0.4];
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
            .expect("a repeated time must still solve");
        assert_eq!(r.n_times, t_eval.len());
    }

    #[test]
    fn a_sensitivity_request_rejects_a_decreasing_t_eval() {
        // Every public entry point validates through the same helper.
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        let err = prepared
            .solve(
                SolveRequest::new(&[0.0, 2.0, 1.0]).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect_err("a decreasing t_eval must be an error on the sensitivity path too");
        assert!(
            matches!(err, CoreError::UnsortedTimePoints { .. }),
            "expected UnsortedTimePoints, got: {err:?}"
        );
    }

    #[test]
    fn solve_y0_length_mismatch_returns_error_not_panic() {
        let model = build_small_dae_with_param(); // n_states = 2
        let (_y0, inputs, t_eval) = small_dae_setup();
        let err = solve(model, &t_eval, &[], &[1.0], &inputs, 1e-8, &[1e-8, 1e-8])
            .expect_err("y0 length mismatch must be a returned error, not a panic");
        assert!(
            matches!(
                err,
                CoreError::Y0Length {
                    got: 1,
                    expected: 2
                }
            ),
            "expected Y0Length {{ got: 1, expected: 2 }}, got: {err:?}"
        );
    }

    #[test]
    fn solve_inputs_length_mismatch_is_a_clear_input_error() {
        let model = build_small_dae_with_param(); // needs 1 packed input
        let (y0, _inputs, t_eval) = small_dae_setup();
        let err = solve(model, &t_eval, &[], &y0, &[], 1e-8, &[1e-8, 1e-8])
            .expect_err("too-short inputs must be an error");
        assert!(
            matches!(
                err,
                CoreError::InputsLength {
                    got: 0,
                    expected: 1
                }
            ),
            "expected InputsLength {{ got: 0, expected: 1 }}, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("inputs") && msg.contains("parameter"),
            "expected a clear inputs-length error, got: {msg}"
        );
    }

    #[test]
    fn prepared_problem_atol_length_mismatch_is_error() {
        let model = build_small_dae_with_param(); // n_states = 2
        let err = PreparedSolver::new(model, 1e-8, &[1e-8])
            .expect_err("atol length mismatch must be a returned error");
        assert!(
            matches!(
                err,
                CoreError::AtolLength {
                    got: 1,
                    expected: 2
                }
            ),
            "expected AtolLength {{ got: 1, expected: 2 }}, got: {err:?}"
        );
    }

    #[test]
    fn equations_implements_implicit_sens_and_solves() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, t_eval) = small_dae_setup();

        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        let eqn = prepared.build_eqn(&y0, &[], &ws, false, &inputs, &prepared.all_param_indices);

        let problem = OdeBuilder::<FaerSparseMat<f64>>::new()
            .rtol(1e-8)
            .atol(vec![1e-8, 1e-8])
            .t0(t_eval[0])
            .p(inputs.clone())
            .build_from_eqn(eqn)
            .expect("build_from_eqn failed");

        let state = problem
            .bdf_state_sens::<ReusedFaerLu>()
            .expect("bdf_state_sens failed");
        let mut solver = problem
            .bdf_solver_sens::<ReusedFaerLu>(state)
            .expect("bdf_solver_sens failed");

        let (_y, ys, _stop) = solver
            .solve_dense_sensitivities(&t_eval)
            .expect("solve_dense_sensitivities failed");

        // One sensitivity matrix per parameter
        assert_eq!(ys.len(), inputs.len());
    }

    #[test]
    fn reused_lu_end_to_end_matches_stock_lu() {
        // Same factorisation algorithm, same arithmetic: swapping the stock
        // solver for the buffer-reusing one must not move a single bit.
        let run = |stock: bool| -> Vec<f64> {
            let model = build_small_dae_with_param();
            let prepared =
                PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
            let (y0, inputs, t_eval) = small_dae_setup();
            let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
            let eqn =
                prepared.build_eqn(&y0, &[], &ws, false, &inputs, &prepared.all_param_indices);
            let problem = OdeBuilder::<FaerSparseMat<f64>>::new()
                .rtol(1e-8)
                .atol(vec![1e-8, 1e-8])
                .t0(t_eval[0])
                .p(inputs)
                .build_from_eqn(eqn)
                .expect("build_from_eqn failed");
            let y_mat = if stock {
                type Stock = diffsol::FaerSparseLU<f64>;
                let state = problem.bdf_state::<Stock>().expect("bdf_state failed");
                let mut solver = problem
                    .bdf_solver::<Stock>(state)
                    .expect("bdf_solver failed");
                solver.solve_dense(&t_eval).expect("solve_dense failed").0
            } else {
                let state = problem
                    .bdf_state::<ReusedFaerLu>()
                    .expect("bdf_state failed");
                let mut solver = problem
                    .bdf_solver::<ReusedFaerLu>(state)
                    .expect("bdf_solver failed");
                solver.solve_dense(&t_eval).expect("solve_dense failed").0
            };
            PreparedSolver::dense_to_column_major(&y_mat)
        };
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn sens_solve_integrates_only_the_requested_column() {
        // Two inputs, sensitivities requested for `a` alone: one integrated column, and
        // it has to be d/da rather than the d/db a mis-mapped subset would hand back.
        let model = build_small_dae_two_inputs_one_sens();
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let y0 = vec![1.0, 0.0];
        let inputs = vec![1.0, 0.5];
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();

        let res = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("sensitivity solve failed");
        assert_eq!(blocks(&res).len(), 1, "2 inputs, 1 requested");

        let h = 1e-6;
        let mut ip = inputs.clone();
        ip[0] += h;
        let rp = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &ip))
            .unwrap();
        let mut im = inputs;
        im[0] -= h;
        let rm = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &im))
            .unwrap();
        let n = res.n_rows;
        let last = (res.n_times - 1) * n;
        for j in 0..n {
            let fd = (rp.y[last + j] - rm.y[last + j]) / (2.0 * h);
            let got = blocks(&res)[0][last + j];
            assert!((got - fd).abs() < 1e-4, "d/da[{j}]: got {got} fd {fd}");
        }
    }

    #[test]
    fn sens_solve_integrates_a_non_prefix_subset_column() {
        // A prefix subset would still pass under a truncating seed; starting past
        // position 0 is what makes the end-to-end solve name d/db rather than d/da.
        let model = build_small_dae_two_inputs(&[1]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let y0 = vec![1.0, 0.0];
        let inputs = vec![1.0, 0.5];
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();

        let res = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("sensitivity solve failed");
        assert_eq!(blocks(&res).len(), 1, "2 inputs, 1 requested");

        let h = 1e-6;
        let mut ip = inputs.clone();
        ip[1] += h;
        let rp = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &ip))
            .unwrap();
        let mut im = inputs;
        im[1] -= h;
        let rm = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &im))
            .unwrap();
        let n = res.n_rows;
        let last = (res.n_times - 1) * n;
        for j in 0..n {
            let fd = (rp.y[last + j] - rm.y[last + j]) / (2.0 * h);
            let got = blocks(&res)[0][last + j];
            assert!((got - fd).abs() < 1e-4, "d/db[{j}]: got {got} fd {fd}");
        }
    }

    #[test]
    fn sens_solve_without_requested_params_is_an_error() {
        // Nothing to differentiate is a caller mistake, not a zero-column solve: the
        // wrapper has no columns to present and the augmented state has no shape.
        let prepared = build_decay_without_sens_params();
        let err = prepared
            .solve(
                SolveRequest::new(&[0.0, 0.1]).with_sensitivities(),
                InputSet::new(&[1.0], &[]),
            )
            .expect_err("a solve with no requested sensitivities must be an error");
        // The config error must surface as-is: routing it through the relaxed
        // retry would double-report it as a misleading SensRetryFailed.
        assert!(
            matches!(err, CoreError::NoSensitivityParams),
            "expected NoSensitivityParams, got: {err:?}"
        );
    }

    /// `dy/dt = -y`, one state, compiled with no sensitivity parameters.
    fn build_decay_without_sens_params() -> PreparedSolver {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs = arena.alloc(Node::Mul(neg, y));
        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(1, 1),
        };
        let model = ModelEvaluator::new(&arena, rhs, mass, 1, 0);
        PreparedSolver::new(model, 1e-8, &[1e-8]).expect("PreparedSolver failed")
    }

    #[test]
    fn a_seed_without_requested_params_reports_the_missing_params_not_its_width() {
        // Expected width is 0 here, so the length complaint ("must be empty or
        // 0") would describe the seed instead of the real mistake.
        let prepared = build_decay_without_sens_params();
        let err = prepared
            .solve(
                SolveRequest::new(&[0.0, 0.1]).with_sensitivities(),
                InputSet::new(&[1.0], &[]).with_sens_seed(&[0.5]),
            )
            .expect_err("a non-empty seed with no requested sensitivities must be an error");
        assert!(
            matches!(err, CoreError::NoSensitivityParams),
            "expected NoSensitivityParams, got: {err:?}"
        );
    }

    #[test]
    fn a_doubly_misconfigured_request_reports_the_seed_before_the_row_space() {
        // One entry point validates for every payload combination, so the order
        // it checks in is a contract: the missing sensitivity parameters are the
        // reason this call cannot run at all, while the absent output variables
        // are only reachable once it can.
        let prepared = build_decay_without_sens_params(); // no sens params, no outputs
        let err = prepared
            .solve(
                SolveRequest::new(&[0.0, 0.1])
                    .with_outputs()
                    .with_sensitivities(),
                InputSet::new(&[1.0], &[]),
            )
            .expect_err("a request for payloads the model cannot supply must be an error");
        assert!(
            matches!(err, CoreError::NoSensitivityParams),
            "expected NoSensitivityParams, got: {err:?}"
        );
    }

    /// `dy/dt = -a*y` (`a = inputs[0]`), output `2*y`, event `y - 0.4`,
    /// sensitivities wrt `a`. Analytic: `y = exp(-a*t)`, `dy/da = -t*exp(-a*t)`.
    fn build_decay_param_output_event() -> PreparedSolver {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let rhs = arena.alloc(Node::Mul(neg_a, sv));
        let two = arena.alloc(Node::Scalar(2.0));
        let output = arena.alloc(Node::Mul(two, sv));
        let threshold = arena.alloc(Node::Scalar(0.4));
        let event = arena.alloc(Node::Sub(sv, threshold));
        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(1, 1),
        };
        let mut model = ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            1,
            1,
            CompiledModelOptions::new().with_sensitivities(&[0]),
        );
        model.add_output(&arena, output);
        model.add_event(&arena, event);
        PreparedSolver::new(model, 1e-10, &[1e-10]).expect("PreparedSolver failed")
    }

    /// Two states, three outputs and one sensitivity parameter, with an event:
    /// the row spaces differ, so a payload reported under the wrong row count is
    /// visible. `store_yp` is on so the `yp` rule has something to report.
    fn build_two_state_three_output_with_sens() -> PreparedSolver {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let two = arena.alloc(Node::Scalar(2.0));
        let neg_two_a = arena.alloc(Node::Mul(neg_a, two));
        let rhs = {
            let d0 = arena.alloc(Node::Mul(neg_a, y0));
            let d1 = arena.alloc(Node::Mul(neg_two_a, y1));
            arena.alloc(Node::Concat(vec![d0, d1]))
        };

        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 1.0],
            shape: Shape::matrix(2, 2),
        };
        let mut model = ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            2,
            1,
            CompiledModelOptions::new().with_sensitivities(&[0]),
        );
        let out0 = arena.alloc(Node::Mul(two, y0));
        let out1 = arena.alloc(Node::Add(y0, y1));
        let out2 = arena.alloc(Node::Mul(two, y1));
        model.add_output(&arena, out0);
        model.add_output(&arena, out1);
        model.add_output(&arena, out2);
        let threshold = arena.alloc(Node::Scalar(0.4));
        let event = arena.alloc(Node::Sub(y0, threshold));
        model.add_event(&arena, event);

        PreparedSolver::new(model, 1e-10, &[1e-10, 1e-10])
            .expect("PreparedSolver failed")
            .with_store_yp(true)
    }

    /// One outcome type means one layout contract, so it can be asserted once
    /// over every payload combination instead of restated per result type — the
    /// prose invariant that the four types stopped honouring when `yp` landed on
    /// two of them.
    #[test]
    fn every_payload_combination_honours_one_layout_contract() {
        let prepared = build_two_state_three_output_with_sens();
        let (n_states, n_outputs) = (2, 3);
        // Runs past the event at y0 = 0.4, so every combination terminates on a
        // root and has a `t_event`/`y_event` to report.
        let t_eval: Vec<f64> = (0..=40).map(|i| f64::from(i) * 0.05).collect();
        let set = InputSet::new(&[1.0, 1.0], &[1.0]);

        for outputs in [false, true] {
            for sensitivities in [false, true] {
                let mut request = SolveRequest::new(&t_eval);
                if outputs {
                    request = request.with_outputs();
                }
                if sensitivities {
                    request = request.with_sensitivities();
                }
                let label = format!("outputs={outputs}, sensitivities={sensitivities}");
                let outcome = prepared.solve(request, set).expect(&label);

                let expected_rows = if outputs { n_outputs } else { n_states };
                assert_eq!(outcome.n_rows, expected_rows, "{label}: row space");
                assert_eq!(outcome.n_times, outcome.t.len(), "{label}: n_times");
                assert_eq!(
                    outcome.y.len(),
                    outcome.n_rows * outcome.n_times,
                    "{label}: trajectory size"
                );

                // yp is the state trajectory's slopes, so it is present exactly
                // when the rows are states and the solver stores them.
                assert_eq!(outcome.yp.is_some(), !outputs, "{label}: yp presence");
                if let Some(yp) = &outcome.yp {
                    assert_eq!(yp.len(), outcome.y.len(), "{label}: yp layout");
                }

                assert_eq!(
                    outcome.sensitivities.is_some(),
                    sensitivities,
                    "{label}: sensitivity presence"
                );
                if let Some(blocks) = &outcome.sensitivities {
                    assert_eq!(blocks.len(), 1, "{label}: one block per parameter");
                    for (i, block) in blocks.iter().enumerate() {
                        assert_eq!(block.len(), outcome.y.len(), "{label}: block {i} layout");
                    }
                }

                assert_eq!(outcome.flag, 1, "{label}: the event should have fired");
                let t_event = outcome.t_event.expect("t_event missing");
                assert!(
                    (t_event - outcome.t[outcome.n_times - 1]).abs() < 1e-12,
                    "{label}: the trajectory should end at the root"
                );
                // Always a full state, never an outputs row: the only thing a
                // caller can restart from.
                let y_event = outcome.y_event.as_ref().expect("y_event missing");
                assert_eq!(y_event.len(), n_states, "{label}: y_event is a full state");
            }
        }
    }

    /// The two requests that differ only in their row space share the
    /// integration, so their termination fields agree bit for bit rather than
    /// merely in shape.
    #[test]
    // Bit-identical is the property under test, so the comparison is exact.
    #[allow(clippy::float_cmp)]
    fn a_row_space_change_leaves_the_termination_fields_untouched() {
        let prepared = build_two_state_three_output_with_sens();
        let t_eval: Vec<f64> = (0..=40).map(|i| f64::from(i) * 0.05).collect();
        let set = InputSet::new(&[1.0, 1.0], &[1.0]);

        let states = prepared
            .solve(SolveRequest::new(&t_eval), set)
            .expect("state solve failed");
        let outputs = prepared
            .solve(SolveRequest::new(&t_eval).with_outputs(), set)
            .expect("output solve failed");

        assert_eq!(states.t, outputs.t);
        assert_eq!(states.n_times, outputs.n_times);
        assert_eq!(states.flag, outputs.flag);
        assert_eq!(states.t_event, outputs.t_event);
        assert_eq!(states.y_event, outputs.y_event);
        assert_eq!(
            states.statistics.number_of_steps,
            outputs.statistics.number_of_steps
        );
    }

    #[test]
    fn an_outputs_sensitivity_request_matches_analytic() {
        // Stops before the event at t = ln 2.5, so this is the final-time path.
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(
                SolveRequest::new(&t_eval)
                    .with_outputs()
                    .with_sensitivities(),
                InputSet::new(&[1.0], &[1.0]),
            )
            .expect("outputs sensitivity solve failed");

        assert_eq!(r.flag, 0);
        assert_eq!(r.n_rows, 1);
        assert_eq!(blocks(&r).len(), 1);
        assert_eq!(blocks(&r)[0].len(), r.n_times);
        for (j, &t) in r.t.iter().enumerate() {
            let out = r.y[j];
            let sens = blocks(&r)[0][j];
            let expected_out = 2.0 * (-t).exp();
            let expected_sens = -2.0 * t * (-t).exp();
            assert!(
                (out - expected_out).abs() < 1e-6,
                "t={t}: out={out}, want {expected_out}"
            );
            assert!(
                (sens - expected_sens).abs() < 1e-4,
                "t={t}: d(out)/da={sens}, want {expected_sens}"
            );
        }
        // Final-time outputs solves must still carry a restartable full state.
        assert!(r.t_event.is_none());
        let y_event = r.y_event.expect("terminal state missing");
        assert_eq!(y_event.len(), 1);
        assert!(
            (y_event[0] - (-0.5f64).exp()).abs() < 1e-6,
            "terminal state {} diverges from exp(-0.5)",
            y_event[0]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: columns keep their times
    fn solve_event_reports_grid_times_then_the_root() {
        // Every column before the root keeps its requested time and the root adds
        // exactly one; nothing is relabelled to the root time or dropped.
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[1.0]))
            .expect("solve failed");

        assert_eq!(r.flag, 1, "expected event termination");
        let t_event = r.t_event.expect("t_event missing");
        let drained = t_eval.iter().filter(|&&t| t <= t_event).count();
        assert_eq!(
            r.n_times,
            drained + 1,
            "one root column past the grid points"
        );
        assert_eq!(r.t.len(), r.n_times);
        assert_eq!(r.n_rows, 1);
        for (j, (&got, &want)) in r.t.iter().zip(&t_eval).take(drained).enumerate() {
            assert_eq!(got, want, "column {j} moved off its requested time");
        }
        assert_eq!(r.t[drained], t_event, "last column is not the root time");
        let y_root = r.y[drained];
        assert!((y_root - 0.4).abs() < 1e-6, "y at the root={y_root}");
        let y_event = r.y_event.expect("y_event missing");
        assert!((y_event[0] - 0.4).abs() < 1e-6, "y_event={}", y_event[0]);
    }

    #[test]
    fn store_yp_fills_the_polynomial_derivative_on_the_state_path() {
        // dy/dt = -a*y with z = y, so every knot's slope is -a*y(t) on both
        // rows; the algebraic row differentiates the same polynomial in z.
        let prepared = PreparedSolver::new(build_small_dae_with_param(), 1e-10, &[1e-10, 1e-10])
            .expect("prepare")
            .with_store_yp(true);
        let (y0, inputs, t_eval) = small_dae_setup();
        let result = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
            .expect("solve");
        let yp = result.yp.expect("yp requested");
        assert_eq!(yp.len(), result.y.len());
        for j in 0..result.n_times {
            let expected = -inputs[0] * result.y[j * result.n_rows];
            for row in 0..result.n_rows {
                let got = yp[j * result.n_rows + row];
                assert!(
                    (got - expected).abs() <= 1e-6 * expected.abs(),
                    "yp[{row}, {j}] = {got} but -a*y = {expected}",
                );
            }
        }
    }

    #[test]
    fn every_solve_path_reports_its_own_integration_time() {
        // All four payload combinations, so a kernel that stopped stamping would
        // report a zero the caller could not tell from a very fast solve.
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.1).collect();
        let statistics = [
            (
                "states",
                prepared
                    .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[1.0]))
                    .expect("solve")
                    .statistics,
            ),
            (
                "outputs",
                prepared
                    .solve(
                        SolveRequest::new(&t_eval).with_outputs(),
                        InputSet::new(&[1.0], &[1.0]),
                    )
                    .expect("output solve")
                    .statistics,
            ),
            (
                "sensitivities",
                prepared
                    .solve(
                        SolveRequest::new(&t_eval).with_sensitivities(),
                        InputSet::new(&[1.0], &[1.0]),
                    )
                    .expect("sensitivity solve")
                    .statistics,
            ),
            (
                "outputs + sensitivities",
                prepared
                    .solve(
                        SolveRequest::new(&t_eval)
                            .with_outputs()
                            .with_sensitivities(),
                        InputSet::new(&[1.0], &[1.0]),
                    )
                    .expect("output sensitivity solve")
                    .statistics,
            ),
        ];

        for (name, statistics) in statistics {
            assert!(
                statistics.integration_time_secs > 0.0,
                "{name} reported no integration time",
            );
            assert!(
                statistics.integration_time_secs
                    >= statistics.ic_time_secs + statistics.solver_setup_time_secs,
                "{name} reported less time than the phases it contains",
            );
        }
    }

    /// A uniform grid of `n` columns over `[0, 1]`, the small-DAE span.
    fn grid(n: usize) -> Vec<f64> {
        assert!(n > 1, "a grid needs at least two columns");
        let last = (n - 1) as f64;
        (0..n).map(|i| i as f64 / last).collect()
    }

    #[test]
    fn a_grid_at_the_hermite_column_limit_still_stores_yp() {
        let prepared = build_small_dae_prepared_with_sens().with_store_yp(true);
        let (y0, inputs, _) = small_dae_setup();
        let result = prepared
            .solve(
                SolveRequest::new(&grid(MAX_HERMITE_COLUMNS)),
                InputSet::new(&y0, &inputs),
            )
            .expect("solve");
        assert_eq!(result.n_times, MAX_HERMITE_COLUMNS);
        assert!(result.yp.is_some(), "the limit itself must keep yp");
    }

    #[test]
    fn a_grid_past_the_hermite_column_limit_gives_up_yp() {
        let prepared = build_small_dae_prepared_with_sens().with_store_yp(true);
        let (y0, inputs, _) = small_dae_setup();
        let columns = MAX_HERMITE_COLUMNS + 1;
        let result = prepared
            .solve(
                SolveRequest::new(&grid(columns)),
                InputSet::new(&y0, &inputs),
            )
            .expect("solve");
        assert_eq!(result.n_times, columns, "every column is still reported");
        assert!(
            result.yp.is_none(),
            "{columns} columns should have dropped yp"
        );
    }

    #[test]
    fn the_hermite_column_limit_applies_to_the_sensitivity_path() {
        let prepared = build_small_dae_prepared_with_sens().with_store_yp(true);
        let (y0, inputs, _) = small_dae_setup();
        let result = prepared
            .solve(
                SolveRequest::new(&grid(MAX_HERMITE_COLUMNS + 1)).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("solve");
        assert!(
            result.yp.is_none(),
            "the sens path drops yp on the same rule"
        );
    }

    #[test]
    fn yp_is_absent_unless_requested() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, t_eval) = small_dae_setup();
        let result = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
            .expect("solve");
        assert!(result.yp.is_none());
    }

    #[test]
    fn the_sensitivity_state_path_stores_yp_too() {
        let prepared = build_small_dae_prepared_with_sens().with_store_yp(true);
        let (y0, inputs, t_eval) = small_dae_setup();
        let result = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("solve");
        let yp = result.yp.expect("yp requested");
        assert_eq!(yp.len(), result.y.len());
        let last = (result.n_times - 1) * result.n_rows;
        let expected = -inputs[0] * result.y[last];
        assert!(
            (yp[last] - expected).abs() <= 1e-6 * expected.abs(),
            "terminal yp = {} but -a*y = {expected}",
            yp[last],
        );
    }

    #[test]
    fn the_root_column_carries_the_wound_back_derivative() {
        // The root column is read off the state after state_mut_back, so its
        // yp must be the root-time slope, not the overshot step's.
        let prepared = build_decay_param_output_event().with_store_yp(true);
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[1.0]))
            .expect("solve failed");
        assert_eq!(r.flag, 1, "expected event termination");
        let yp = r.yp.expect("yp requested");
        let root = r.n_times - 1;
        assert!(
            (yp[root] - (-0.4)).abs() < 1e-6,
            "yp at the y = 0.4 root is {}, want -0.4",
            yp[root],
        );
    }

    #[test]
    fn an_outputs_sensitivity_request_ends_at_the_event_root() {
        // Runs past the event at y = 0.4 (t = ln 2.5): the root must end the
        // trajectory and the sensitivity columns must span the same times.
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(
                SolveRequest::new(&t_eval)
                    .with_outputs()
                    .with_sensitivities(),
                InputSet::new(&[1.0], &[1.0]),
            )
            .expect("outputs sensitivity solve failed");

        assert_eq!(r.flag, 1, "expected event termination");
        let t_event = r.t_event.expect("t_event missing");
        let ln2p5 = 2.5f64.ln();
        assert!(
            (t_event - ln2p5).abs() < 1e-6,
            "t_event={t_event}, want {ln2p5}"
        );
        assert_eq!(r.n_times, r.t.len());
        assert_eq!(blocks(&r)[0].len(), r.n_times);
        // y_event is the full state at the root, not the outputs row.
        let y_event = r.y_event.as_deref().expect("y_event missing");
        assert!(
            (y_event[0] - 0.4).abs() < 1e-6,
            "y_event={}, expected state 0.4",
            y_event[0]
        );
        let out_last = r.y[r.n_times - 1];
        assert!((out_last - 0.8).abs() < 1e-6, "out at root={out_last}");
        let sens_last = blocks(&r)[0][r.n_times - 1];
        let expected = -2.0 * t_event * 0.4;
        assert!(
            (sens_last - expected).abs() < 1e-4,
            "d(out)/da at root={sens_last}, want {expected}"
        );
    }

    /// `(ws, eqn)` over the two-input fixture with the given sensitivity subset, so a
    /// parameter-indexed `f_p` is distinguishable from a subset-indexed one.
    fn two_input_rhs_at_y0(sens: &[usize]) -> (Rc<RefCell<Workspace>>, Equations) {
        let model = build_small_dae_two_inputs(sens);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        assert_eq!(prepared.compiled.n_sens_params(), sens.len());
        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        // a = 1.0, b = 0.5, y0 = [1, 0]
        let eqn = prepared.build_eqn(
            &[1.0, 0.0],
            &[],
            &ws,
            false,
            &[1.0, 0.5],
            &prepared.sens_param_indices,
        );
        (ws, eqn)
    }

    /// Collect the dense `n x ncols` matrix `m` in column-major order.
    fn dense_columns(m: &FaerSparseMat<f64>, n: usize, ncols: usize) -> Vec<f64> {
        let mut got = vec![0.0; n * ncols];
        let (indices, values) = m.triplet_iter();
        for ((i, j), val) in indices.zip(values) {
            got[j * n + i] = val;
        }
        got
    }

    #[test]
    fn rhs_sens_inplace_assembles_the_requested_subset_in_order() {
        // Reordered subset [b, a]: column order follows the subset, not the input
        // vector, so a mis-mapped scatter shows up as swapped columns.
        let (_ws, eqn) = two_input_rhs_at_y0(&[1, 0]);
        let rhs = eqn.rhs();
        let ctx = *rhs.context();
        let mut x = FaerVec::<f64>::zeros(2, ctx);
        x.set_index(0, 1.0);
        let (n, np) = (rhs.nout(), rhs.nparams());
        assert_eq!(np, 2);

        let mut m = FaerSparseMat::<f64>::new_from_sparsity(n, np, rhs.sens_sparsity(), ctx);
        rhs.sens_inplace(&x, 0.0, &mut m);

        // f = [-a*y + b, y - z], so df/db = [1, 0] and df/da = [-y, 0] = [-1, 0].
        assert_eq!(dense_columns(&m, n, np), vec![1.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    fn rhs_sens_inplace_narrows_to_the_requested_column() {
        // Subset [b] alone: one column, and it must be d/db. A prefix-truncating
        // seed would hand back d/da instead.
        let (_ws, eqn) = two_input_rhs_at_y0(&[1]);
        let rhs = eqn.rhs();
        let ctx = *rhs.context();
        let mut x = FaerVec::<f64>::zeros(2, ctx);
        x.set_index(0, 1.0);
        let (n, np) = (rhs.nout(), rhs.nparams());
        assert_eq!(np, 1, "2 inputs, 1 requested");

        let mut m = FaerSparseMat::<f64>::new_from_sparsity(n, np, rhs.sens_sparsity(), ctx);
        rhs.sens_inplace(&x, 0.0, &mut m);

        assert_eq!(dense_columns(&m, n, np), vec![1.0, 0.0]);
    }

    #[test]
    fn rhs_sens_inplace_matches_per_column_sens_mul_on_a_subset() {
        // Sharing one primal pass must not move the answer, including for a
        // selection that is reordered relative to the input vector.
        let (_ws, eqn) = two_input_rhs_at_y0(&[1, 0]);
        let rhs = eqn.rhs();
        let ctx = *rhs.context();
        let mut x = FaerVec::<f64>::zeros(2, ctx);
        x.set_index(0, 1.0);
        let (n, np) = (rhs.nout(), rhs.nparams());

        let mut expected = vec![0.0; n * np];
        let mut v = FaerVec::<f64>::zeros(np, ctx);
        let mut col = FaerVec::<f64>::zeros(n, ctx);
        for j in 0..np {
            v.set_index(j, 1.0);
            rhs.sens_mul_inplace(&x, 0.0, &v, &mut col);
            expected[j * n..(j + 1) * n].copy_from_slice(col.as_slice());
            v.set_index(j, 0.0);
        }

        let mut m = FaerSparseMat::<f64>::new_from_sparsity(n, np, rhs.sens_sparsity(), ctx);
        rhs.sens_inplace(&x, 0.0, &mut m);

        assert_eq!(dense_columns(&m, n, np), expected);
    }

    #[test]
    fn sens_inplace_runs_one_primal_pass_regardless_of_k() {
        // The batching mechanism, not a timing: one shared primal pass feeds every
        // requested column, at both k = 1 and k = 2.
        for sens in [&[0][..], &[1, 0][..]] {
            let (ws, eqn) = two_input_rhs_at_y0(sens);
            let rhs = eqn.rhs();
            let ctx = *rhs.context();
            let mut x = FaerVec::<f64>::zeros(2, ctx);
            x.set_index(0, 1.0);

            ws.borrow_mut().sens_primal_passes = 0;
            let (n, np) = (rhs.nout(), rhs.nparams());
            assert_eq!(np, sens.len(), "subset width should equal k");
            let mut m = FaerSparseMat::<f64>::new_from_sparsity(n, np, rhs.sens_sparsity(), ctx);
            rhs.sens_inplace(&x, 0.0, &mut m);

            assert_eq!(ws.borrow().sens_primal_passes, 1, "k = {np}");
        }
    }

    #[test]
    fn set_params_splices_only_the_sensitivity_slots() {
        // k values in, spliced at their global indices: the surplus input keeps
        // the value the solve was handed.
        let model = build_small_dae_two_inputs(&[1]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        let mut eqn = prepared.build_eqn(
            &[1.0, 0.0],
            &[],
            &ws,
            false,
            &[1.0, 0.5],
            &prepared.sens_param_indices,
        );
        assert_eq!(eqn.nparams(), 1);

        let ctx = *eqn.context();
        let mut p = FaerVec::<f64>::zeros(1, ctx);
        p.set_index(0, 7.0);
        eqn.set_params(&p);

        assert_eq!(eqn.params.as_slice(), &[1.0, 7.0]);
    }

    #[test]
    fn get_params_round_trips_the_subset() {
        let model = build_small_dae_two_inputs(&[1]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        let eqn = prepared.build_eqn(
            &[1.0, 0.0],
            &[],
            &ws,
            false,
            &[1.0, 0.5],
            &prepared.sens_param_indices,
        );

        let ctx = *eqn.context();
        let mut p = FaerVec::<f64>::zeros(eqn.nparams(), ctx);
        eqn.get_params(&mut p);

        assert_eq!(p.as_slice(), &[0.5]);
    }

    #[test]
    fn plain_solve_of_a_sens_model_takes_the_full_input_vector() {
        // The identity subset on the plain path: a model compiled with a
        // one-parameter subset still solves against both inputs.
        let model = build_small_dae_two_inputs(&[1]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();

        let res = prepared
            .solve(
                SolveRequest::new(&t_eval),
                InputSet::new(&[1.0, 0.0], &[1.0, 0.5]),
            )
            .expect("plain solve of a sens-compiled model must accept the full input vector");

        // dy/dt = -a*y + b with a = 1, b = 0.5 relaxes towards b/a = 0.5.
        let last = (res.n_times - 1) * res.n_rows;
        assert!(
            res.y[last] > 0.5 && res.y[last] < 1.0,
            "y = {}",
            res.y[last]
        );
    }

    #[test]
    fn seeded_y0_sens_matches_finite_difference_through_the_initial_condition() {
        // y0 itself is `a`, so d/da picks up the seed as well as the rhs term.
        // Zeroing the seed (the old behaviour) misses the first contribution.
        let prepared = build_small_dae_prepared_with_sens();
        let (_, inputs, t_eval) = small_dae_setup();
        let a = inputs[0];
        let y0 = vec![a, a];
        let seed = [1.0, 1.0];

        let res = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs).with_sens_seed(&seed),
            )
            .expect("seeded sensitivity solve failed");

        let h = 1e-6;
        let rp = prepared
            .solve(
                SolveRequest::new(&t_eval),
                InputSet::new(&[a + h, a + h], &[a + h]),
            )
            .expect("perturbed solve failed");
        let rm = prepared
            .solve(
                SolveRequest::new(&t_eval),
                InputSet::new(&[a - h, a - h], &[a - h]),
            )
            .expect("perturbed solve failed");
        let n = res.n_rows;
        let last = (res.n_times - 1) * n;
        for j in 0..n {
            let fd = (rp.y[last + j] - rm.y[last + j]) / (2.0 * h);
            let got = blocks(&res)[0][last + j];
            assert!((got - fd).abs() < 1e-4, "d/da[{j}]: got {got} fd {fd}");
        }

        let unseeded = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("unseeded sensitivity solve failed");
        assert!(
            (blocks(&unseeded)[0][last] - blocks(&res)[0][last]).abs() > 1e-3,
            "the seed has to move the answer, else the test proves nothing",
        );
    }

    #[test]
    fn a_wrongly_sized_y0_sens_is_an_error_not_a_panic() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, t_eval) = small_dae_setup();
        let err = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs).with_sens_seed(&[1.0]),
            )
            .expect_err("a 1-entry seed for a 2-state model must be rejected");
        assert!(matches!(
            err,
            CoreError::Y0SensLength {
                got: 1,
                expected: 2
            }
        ));
    }

    #[test]
    fn a_sensitivity_request_matches_finite_difference_on_inputs() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, t_eval) = small_dae_setup();
        let res = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .unwrap();
        assert_eq!(blocks(&res).len(), inputs.len());
        // Finite-difference dy/dp0 at the last time point vs the sens-block tail.
        let h = 1e-6;
        let mut ip = inputs.clone();
        ip[0] += h;
        let rp = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &ip))
            .unwrap();
        let mut im = inputs; // last user of inputs; move instead of clone
        im[0] -= h;
        let rm = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &im))
            .unwrap();
        let n = res.n_rows;
        let last = (res.n_times - 1) * n;
        for j in 0..n {
            let fd = (rp.y[last + j] - rm.y[last + j]) / (2.0 * h);
            let got = blocks(&res)[0][last + j];
            assert!(
                (got - fd).abs() < 1e-4,
                "sensitivity block 0, entry {j}: got {got} fd {fd}"
            );
        }
    }

    #[test]
    fn algebraic_rows_are_detected_from_the_mass_matrix() {
        // build_small_dae_two_inputs has mass row 0 = [1.0] and row 1 empty.
        let model = build_small_dae_two_inputs(&[0]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        assert_eq!(prepared.algebraic_rows(), &[false, true]);
    }

    #[test]
    fn sens_atol_tightens_only_differential_rows() {
        let model = build_small_dae_two_inputs(&[0]);
        let prepared = PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8])
            .expect("PreparedSolver failed")
            .with_sens_atol_factor(1e-3)
            .expect("factor rejected");
        let sens_atol = prepared.sens_atol();
        assert!(
            (sens_atol[0] / 1e-11 - 1.0).abs() < 1e-12,
            "differential row was not tightened: {}",
            sens_atol[0]
        );
        assert!(
            (sens_atol[1] / 1e-8 - 1.0).abs() < 1e-12,
            "algebraic row must keep the state atol: {}",
            sens_atol[1]
        );
    }

    #[test]
    fn default_sens_atol_factor_tightens_differential_rows() {
        let model = build_small_dae_two_inputs(&[0]);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let sens_atol = prepared.sens_atol();
        assert!(
            (sens_atol[0] / (1e-8 * DEFAULT_SENS_ATOL_FACTOR) - 1.0).abs() < 1e-12,
            "default factor not applied: {}",
            sens_atol[0]
        );
    }

    #[test]
    fn with_sens_atol_factor_rejects_non_positive_and_non_finite() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let model = build_small_dae_two_inputs(&[0]);
            let prepared =
                PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
            assert!(
                prepared.with_sens_atol_factor(bad).is_err(),
                "factor {bad} should be rejected"
            );
        }
    }

    #[test]
    fn successful_sens_solve_does_not_report_a_relaxed_error_control() {
        let model = build_small_dae_two_inputs_one_sens();
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();
        let result = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&[1.0, 0.0], &[1.0, 0.5]),
            )
            .expect("sens solve failed");
        assert!(!result.statistics.sens_error_control_relaxed);
    }

    #[test]
    fn relaxed_error_control_still_integrates_the_sensitivity() {
        // The retry path must produce the same answer, just under looser control.
        let prepared = || {
            PreparedSolver::new(
                build_small_dae_two_inputs_one_sens(),
                1e-10,
                &[1e-10, 1e-10],
            )
            .expect("PreparedSolver failed")
        };
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();
        let (y0, inputs) = ([1.0, 0.0], [1.0, 0.5]);

        let controlled = prepared()
            .run_dense_sensitivities_inner(
                TimePlan::new(&t_eval, &[]),
                &y0,
                &[],
                &inputs,
                false,
                true,
            )
            .expect("controlled solve failed");
        let relaxed = prepared()
            .run_dense_sensitivities_inner(
                TimePlan::new(&t_eval, &[]),
                &y0,
                &[],
                &inputs,
                false,
                false,
            )
            .expect("relaxed solve failed");

        let last = t_eval.len() - 1;
        let controlled_s = controlled.1[0][last * controlled.0.n_rows];
        let relaxed_s = relaxed.1[0][last * relaxed.0.n_rows];
        assert!(
            (controlled_s - relaxed_s).abs() < 1e-5 * controlled_s.abs().max(1e-8),
            "relaxed sens {relaxed_s} diverged from controlled {controlled_s}"
        );
    }
}
