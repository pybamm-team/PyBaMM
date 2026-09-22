//! Problem setup and the solve loop.
//!
//! [`PreparedSolver`] holds everything reusable about a model; each solve builds its own
//! workspace, problem and diffsol instance, keeping the handle `Send + Sync`.

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::ode_solver::OdeSolverStatistics;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{
    AugmentedOdeSolverMethod, BacktrackingLineSearch, BdfState, FaerContext, FaerSparseLU,
    InitialConditionSolverOptions, NewtonNonlinearSolver, NoLineSearch, NonLinearOp, OdeBuilder,
    OdeEquations, OdeSolverMethod, OdeSolverProblem, OdeSolverStopReason, Op, SensEquations,
    StateRef, StateRefMut, Vector,
};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use faer::{Mat, MatMut, Par};

use super::equations::Equations;
use super::options::SolverOptions;
use super::{FaerSparsity, csc_to_faer_sparsity, csr_mass_to_faer_csc, lane, lane_mut, lanes};
use crate::arena::{Arena, DagRef, NodeId};
use crate::error::CoreError;
use crate::eval::CompiledExpr;
use crate::jacobian::CscPattern;
use crate::model::{CompiledModel, Workspace};
use crate::node::{CsrData, Node};
use crate::params::ParamBinding;
use crate::simplify::{dce, simplify_pipeline};
use crate::tangent::derivative_wrt_time;

/// Sparse LU used by every Newton iteration.
type SparseLu = FaerSparseLU<f64>;

/// Default multiplier on differential-state `atol` for the sensitivity `atol` floor.
///
/// 1.0 matches IDAS's `IDASensEEtolerances`: `atol / |p_j|`, with `sens_rtol = rtol`.
pub const DEFAULT_SENS_ATOL_FACTOR: f64 = 1.0;

/// Solver statistics from a BDF solve.
#[derive(Debug, Clone, Default)]
pub struct SolverStatistics {
    /// diffsol's step, Newton and linear-solver counters.
    pub solver: OdeSolverStatistics,
    /// Time spent building this solve's workspace, `Equations` and problem.
    pub problem_setup_time_secs: f64,
    /// Time spent computing consistent initial conditions.
    pub ic_time_secs: f64,
    /// Time spent building the BDF solver around the consistent initial state.
    pub solver_setup_time_secs: f64,
    /// Wall-clock time for the whole solve, excluding FFI marshalling.
    pub integration_time_secs: f64,
    /// True when the sensitivity solve was retried with sensitivities out of error control.
    pub sens_error_control_relaxed: bool,
}

/// Seconds since `mark`, moving `mark` to now.
fn lap(mark: &mut Instant) -> f64 {
    let now = Instant::now();
    let secs = now.duration_since(*mark).as_secs_f64();
    *mark = now;
    secs
}

/// How a solve ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Termination {
    Completed,
    /// An event root stopped the solve at this time.
    Event {
        t: f64,
    },
}

impl Termination {
    /// The flag the solver boundary carries: `0` completed, `1` event root.
    pub const fn as_flag(self) -> i32 {
        match self {
            Self::Completed => 0,
            Self::Event { .. } => 1,
        }
    }

    pub const fn event_time(self) -> Option<f64> {
        match self {
            Self::Completed => None,
            Self::Event { t } => Some(t),
        }
    }
}

/// What one solve returns, whichever payloads were asked for.
///
/// On an event root, `t` ends at the root time rather than the last requested point.
#[derive(Debug)]
pub struct SolveOutcome {
    /// Output times, one per trajectory column.
    pub t: Vec<f64>,
    /// Flat column-major trajectory: `y[i + j * n_rows]` is row `i` at time `j`.
    /// Rows are states, or output variables when [`SolveRequest::outputs`] is set.
    pub y: Vec<f64>,
    /// Rows in `y`, and in each sensitivity block.
    pub n_rows: usize,
    /// Row time derivatives in `y`'s layout: only on a state trajectory with
    /// [`PreparedSolver::with_store_yp`] set and at most `MAX_HERMITE_COLUMNS` columns.
    pub yp: Option<Vec<f64>>,
    /// One block per sensitivity parameter in `y`'s layout and request order.
    /// On an outputs request each block is the full `dg/dp + dg/dy · y_s`.
    pub sensitivities: Option<Vec<Vec<f64>>>,
    /// The full state at a root, or at the end on an outputs request; never an outputs row.
    pub y_event: Option<Vec<f64>>,
    pub termination: Termination,
    /// The state after the initial-condition solve, consistent even if `y0` was a guess.
    pub y_start: Vec<f64>,
    /// State sensitivities at `y_start`, one `n_states` block per sensitivity parameter.
    pub y_start_sens: Option<Vec<Vec<f64>>>,
    /// State sensitivities at `y_event`, laid out as `y_start_sens`.
    pub y_event_sens: Option<Vec<Vec<f64>>>,
    pub statistics: SolverStatistics,
}

/// Copy every lane of a batched sensitivity vector, one block per parameter.
fn sens_blocks(s: &FaerVec<f64>) -> Vec<Vec<f64>> {
    lanes(s).map(<[f64]>::to_vec).collect()
}

/// What to integrate over and which payloads to report.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SolveRequest<'a> {
    pub t_eval: &'a [f64],
    /// Discontinuity times the integrator lands on and restarts from.
    /// Entries not also in `t_eval` are integrated through.
    pub t_stop: &'a [f64],
    /// Report the model's output variables instead of the full state.
    pub outputs: bool,
    /// Solve the forward-sensitivity system, seeded per set by [`InputSet::y0_sens`].
    pub sensitivities: bool,
    /// Workers of the current rayon pool the sparse LU may use; 1 factorises serially.
    pub lu_threads: usize,
}

impl<'a> SolveRequest<'a> {
    /// A state-trajectory request over `t_eval`: no stop times, no sensitivities.
    pub const fn new(t_eval: &'a [f64]) -> Self {
        Self {
            t_eval,
            t_stop: &[],
            outputs: false,
            sensitivities: false,
            lu_threads: 1,
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

    /// Factorise with up to `threads` workers of the current rayon pool.
    pub const fn with_lu_threads(mut self, threads: usize) -> Self {
        self.lu_threads = threads;
        self
    }

    fn lu_par(&self) -> Par {
        if self.lu_threads > 1 {
            Par::rayon(self.lu_threads)
        } else {
            Par::Seq
        }
    }
}

/// One input set of a solve: where the state starts and what the parameters are.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct InputSet<'a> {
    pub y0: &'a [f64],
    /// Flat input-parameter vector.
    pub inputs: &'a [f64],
    /// `dy0/dp`, column-major `n_states x k` over the sensitivity subset; empty is zero.
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
/// Holds no per-solve mutable state, so it is `Send + Sync`.
pub struct PreparedSolver {
    compiled: Arc<CompiledModel>,
    rtol: f64,
    atol: Vec<f64>,
    /// Multiplier applied to `atol` on differential states only.
    sens_atol_factor: f64,
    options: SolverOptions,
    store_yp: bool,
    /// `dF/dt` at fixed states and inputs, for time-dependent algebraic constraints.
    time_derivative: Option<CompiledExpr>,
    /// How a pure ODE reads `dy/dt` off `f`; `None` on a DAE.
    ode_mass: Option<OdeMass>,
    /// `g_v`'s layout, built once [`Self::with_store_yp`] needs it on a DAE.
    algebraic_block: Option<AlgebraicBlock>,
    jac_sparsity: Arc<FaerSparsity>,
    mass_sparsity: Arc<FaerSparsity>,
    /// M's values in `mass_sparsity` order.
    mass_csc_values: Arc<[f64]>,
    /// The configured sensitivity subset.
    sens_binding: ParamBinding,
}

impl std::fmt::Debug for PreparedSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSolver")
            .field("n_states", &self.compiled.n_states())
            .field("input_width", &self.compiled.input_width())
            .field("n_events", &self.compiled.n_events())
            .field("n_outputs", &self.compiled.total_output_len())
            .finish_non_exhaustive()
    }
}

impl PreparedSolver {
    /// Build a prepared problem from a compiled model and tolerances.
    pub fn new(
        model: impl Into<Arc<CompiledModel>>,
        rtol: f64,
        atol: &[f64],
    ) -> Result<Self, CoreError> {
        let compiled = model.into();
        let n_states = compiled.n_states();
        if atol.len() != n_states {
            return Err(CoreError::AtolLength {
                got: atol.len(),
                expected: n_states,
            });
        }
        validate_tolerances(rtol, atol)?;

        let mass = compiled.mass_matrix();
        let (mass_sparsity, mass_csc_values) = csr_mass_to_faer_csc(mass);
        let ode_mass = (compiled.state_kinds().n_algebraic() == 0)
            .then(|| OdeMass::new(mass, &mass_sparsity, &mass_csc_values))
            .flatten();
        let sens_binding =
            ParamBinding::sensitivities(compiled.params(), compiled.sens_param_indices());

        Ok(Self {
            jac_sparsity: Arc::new(csc_to_faer_sparsity(compiled.csc_sparsity())),
            compiled,
            rtol,
            atol: atol.to_vec(),
            sens_atol_factor: DEFAULT_SENS_ATOL_FACTOR,
            options: SolverOptions::default(),
            store_yp: false,
            time_derivative: None,
            ode_mass,
            algebraic_block: None,
            mass_sparsity: Arc::new(mass_sparsity),
            mass_csc_values: mass_csc_values.into(),
            sens_binding,
        })
    }

    fn n_states(&self) -> usize {
        self.compiled.n_states()
    }

    fn n_outputs(&self) -> usize {
        self.compiled.total_output_len()
    }

    /// The plain path's binding: the packed input array itself.
    fn packed_binding(&self) -> ParamBinding {
        ParamBinding::packed_inputs(self.compiled.input_width())
    }

    fn build_eqn(
        &self,
        set: InputSet<'_>,
        ws: &Rc<RefCell<Workspace>>,
        binding: &ParamBinding,
        lu_par: Par,
    ) -> Equations {
        let mut context = FaerContext::default();
        context.par = lu_par;
        Equations {
            compiled: Arc::clone(&self.compiled),
            ws: Rc::clone(ws),
            params: set.inputs.to_vec(),
            binding: binding.clone(),
            y0: set.y0.to_vec(),
            y0_sens: set.y0_sens.to_vec(),
            jac_sparsity: Arc::clone(&self.jac_sparsity),
            mass_sparsity: Arc::clone(&self.mass_sparsity),
            mass_csc_values: Arc::clone(&self.mass_csc_values),
            context,
        }
    }

    /// One solve's workspace and diffsol problem over `binding`'s parameter vector.
    ///
    /// `sens_error_control` puts the sensitivities under error control, scaled by `|p_j|`.
    fn build_problem(
        &self,
        set: InputSet<'_>,
        t0: f64,
        binding: &ParamBinding,
        sens_error_control: bool,
        lu_par: Par,
    ) -> Result<(Rc<RefCell<Workspace>>, OdeSolverProblem<Equations>), CoreError> {
        let ws = Rc::new(RefCell::new(self.compiled.create_workspace()));
        let eqn = self.build_eqn(set, &ws, binding, lu_par);
        // The ops report the binding's width, so p narrows with it.
        let mut p = vec![0.0; binding.len()];
        binding.gather_from(set.inputs, &mut p);
        let mut builder = OdeBuilder::<FaerSparseMat<f64>>::new()
            .rtol(self.rtol)
            .atol(self.atol.clone());
        if sens_error_control {
            // param_scales plays the role of IDAS's pbar.
            builder = builder
                .sens_rtol(self.rtol)
                .sens_atol(self.sens_atol())
                .param_scales(Self::param_scales(&p));
        }
        let problem = self.options.apply(builder.t0(t0).p(p).build_from_eqn(eqn)?);
        Ok((ws, problem))
    }

    /// Set the multiplier on differential-state `atol` for the sensitivity tolerance floor.
    pub fn with_sens_atol_factor(mut self, factor: f64) -> Result<Self, CoreError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CoreError::SensAtolFactor { got: factor });
        }
        self.sens_atol_factor = factor;
        Ok(self)
    }

    /// Set the integrator tuning applied to every problem this template builds.
    pub fn with_options(mut self, options: SolverOptions) -> Result<Self, CoreError> {
        options.validate()?;
        self.options = options;
        Ok(self)
    }

    /// Store state time derivatives (`yp`) alongside `y` on state-trajectory paths.
    ///
    /// Doubles trajectory memory, so it is opt-in.
    #[must_use]
    pub fn with_store_yp(mut self, store_yp: bool) -> Self {
        self.store_yp = store_yp;
        let has_algebraic = self.compiled.state_kinds().n_algebraic() > 0;
        if store_yp && has_algebraic && self.algebraic_block.is_none() {
            let block = AlgebraicBlock::new(self.compiled.csc_sparsity(), self.algebraic_states());
            self.algebraic_block = Some(block);
        }
        self
    }

    /// Compile `dF/dt` from `rhs`, the root this model was compiled from in `arena`.
    ///
    /// Only compiled when there are algebraic states and `rhs` reads `t`.
    #[must_use]
    pub fn with_time_derivative(mut self, arena: &Arena, rhs: NodeId) -> Self {
        let reads_time = || {
            arena
                .topological_order(rhs)
                .iter()
                .any(|&id| matches!(arena.get(id), Node::Time))
        };
        if self.compiled.state_kinds().n_algebraic() > 0 && reads_time() {
            // Copies only what `rhs` reaches, not the whole graph.
            let mut derivative = dce(DagRef::new(arena, rhs));
            derivative.rewrite_in_place(derivative_wrt_time);
            self.time_derivative = Some(CompiledExpr::new(simplify_pipeline(derivative).as_ref()));
        }
        self
    }

    /// Whether a solve reporting `n_columns` columns stores `yp` alongside `y`.
    const fn store_yp(&self, n_columns: usize) -> bool {
        self.store_yp && n_columns <= MAX_HERMITE_COLUMNS
    }

    fn algebraic_states(&self) -> &[bool] {
        self.compiled.state_kinds().algebraic_mask()
    }

    /// Per-state sensitivity absolute tolerance.
    ///
    /// Algebraic states keep the state `atol`: tightening them fails DAE sensitivity solves.
    fn sens_atol(&self) -> Vec<f64> {
        self.atol
            .iter()
            .zip(self.algebraic_states())
            .map(|(a, is_algebraic)| {
                if *is_algebraic {
                    *a
                } else {
                    a * self.sens_atol_factor
                }
            })
            .collect()
    }

    /// Per-parameter scales `|p_j|` for the sensitivity absolute tolerances.
    ///
    /// `sens_inputs` is the length-`k` sensitivity subset, not the packed inputs; zero
    /// or non-finite values fall back to 1.0, since diffsol rejects them.
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

    /// An empty trajectory over `n_points` columns of states, or of outputs when `with_output`.
    fn empty_trajectory(
        &self,
        n_points: usize,
        with_output: bool,
        y_start: &[f64],
    ) -> DenseTrajectory {
        // Output mode never stores yp: its trajectory holds outputs, not states.
        let store_yp = self.store_yp(n_points) && !with_output;
        let n_rows = if with_output {
            self.n_outputs()
        } else {
            self.n_states()
        };
        let mut trajectory = DenseTrajectory::with_capacity(n_rows, n_points, store_yp);
        trajectory.y_start = y_start.to_vec();
        trajectory
    }

    /// Validate caller-supplied solve arguments against the model dimensions.
    fn validate_args(&self, y0: &[f64], inputs: &[f64]) -> Result<(), CoreError> {
        if y0.len() != self.n_states() {
            return Err(CoreError::Y0Length {
                got: y0.len(),
                expected: self.n_states(),
            });
        }
        let width = self.compiled.input_width().get();
        if inputs.len() != width {
            return Err(CoreError::InputsLength {
                got: inputs.len(),
                expected: width,
            });
        }
        Ok(())
    }

    fn validate_has_outputs(&self) -> Result<(), CoreError> {
        if self.n_outputs() == 0 {
            return Err(CoreError::NoOutputVariables);
        }
        Ok(())
    }

    /// Validate the caller-supplied `dy0/dp` seed: empty, or column-major `n_states x k`.
    fn validate_y0_sens(&self, y0_sens: &[f64]) -> Result<(), CoreError> {
        if self.sens_binding.is_empty() {
            return Err(CoreError::NoSensitivityParams);
        }
        let expected = self.n_states() * self.sens_binding.len();
        if y0_sens.is_empty() || y0_sens.len() == expected {
            return Ok(());
        }
        Err(CoreError::Y0SensLength {
            got: y0_sens.len(),
            expected,
        })
    }

    /// Integrate the model, returning whichever payloads `request` asked for.
    pub fn solve(
        &self,
        request: SolveRequest<'_>,
        set: InputSet<'_>,
    ) -> Result<SolveOutcome, CoreError> {
        let times = TimePlan::new(request.t_eval, request.t_stop)?;
        self.validate_args(set.y0, set.inputs)?;
        if request.sensitivities {
            self.validate_y0_sens(set.y0_sens)?;
        }
        if request.outputs {
            self.validate_has_outputs()?;
        }
        let lu_par = request.lu_par();

        if request.sensitivities {
            self.solve_sensitivities(times, set, request.outputs, lu_par)
        } else {
            self.solve_states(times, set, request.outputs, lu_par)
        }
    }

    /// Integrate without sensitivities; reports output rows when `with_output` is set.
    fn solve_states(
        &self,
        times: TimePlan<'_>,
        set: InputSet<'_>,
        with_output: bool,
        lu_par: Par,
    ) -> Result<SolveOutcome, CoreError> {
        let started = Instant::now();
        let mut mark = started;
        let mut statistics = SolverStatistics::default();
        // Nothing here reads a `dy0/dp` seed.
        let set = InputSet::new(set.y0, set.inputs);
        let (ws, problem) =
            self.build_problem(set, times.t0(), &self.packed_binding(), false, lu_par)?;
        statistics.problem_setup_time_secs = lap(&mut mark);
        let state = problem.bdf_state::<SparseLu>()?;
        statistics.ic_time_secs = lap(&mut mark);
        let mut solver = problem.bdf_solver::<SparseLu>(state)?;
        statistics.solver_setup_time_secs = lap(&mut mark);

        let run = Run {
            times,
            ws: &ws,
            inputs: set.inputs,
            with_output,
            started,
        };
        let mut recorder = Recorder::new(self, run, false, &solver.state());
        let mut restarter = BreakpointRestarter::new(self, &problem.ic_options);
        let root = drive_segments(&mut solver, times, &mut recorder, |solver, t_next| {
            restarter.restart(solver, t_next)
        })?;
        statistics.solver = solver.get_statistics().clone();
        Ok(recorder.finish(root, &solver.state(), statistics))
    }

    /// Integrate with sensitivities, retrying once without them in error control.
    fn solve_sensitivities(
        &self,
        times: TimePlan<'_>,
        set: InputSet<'_>,
        with_output: bool,
        lu_par: Par,
    ) -> Result<SolveOutcome, CoreError> {
        let started = Instant::now();
        let mut mark = started;
        let mut statistics = SolverStatistics::default();
        let (ws, mut problem) =
            self.build_problem(set, times.t0(), &self.sens_binding, true, lu_par)?;
        statistics.problem_setup_time_secs = lap(&mut mark);
        // Only the state tolerances shape the IC, so both attempts start from this state.
        let state = problem.bdf_state_sens::<SparseLu>()?;
        statistics.ic_time_secs = lap(&mut mark);

        let run = Run {
            times,
            ws: &ws,
            inputs: set.inputs,
            with_output,
            started,
        };
        match self.integrate_sensitivities(&problem, state.clone(), run, statistics.clone()) {
            // Only an integration failure depends on the sensitivities' error control.
            Err(CoreError::Diffsol(controlled)) => {
                // `SensEquations` reads these as the solver is built, so this relaxes it.
                problem.sens_rtol = None;
                problem.sens_atol = None;
                let mut outcome = self
                    .integrate_sensitivities(&problem, state, run, statistics)
                    .map_err(|relaxed| CoreError::SensRetryFailed {
                        controlled: controlled.to_string(),
                        relaxed: relaxed.to_string(),
                    })?;
                outcome.statistics.sens_error_control_relaxed = true;
                Ok(outcome)
            },
            other => other,
        }
    }

    /// One sensitivity integration from `state`, under `problem`'s sensitivity tolerances.
    ///
    /// `statistics` carries the setup times the attempts share.
    fn integrate_sensitivities(
        &self,
        problem: &OdeSolverProblem<Equations>,
        state: BdfState<FaerVec<f64>>,
        run: Run<'_>,
        mut statistics: SolverStatistics,
    ) -> Result<SolveOutcome, CoreError> {
        let mut mark = Instant::now();
        let mut solver = problem.bdf_solver_sens::<SparseLu>(state)?;
        statistics.solver_setup_time_secs = lap(&mut mark);
        let mut recorder = Recorder::new(self, run, true, &solver.state());
        let mut restarter = BreakpointRestarter::new(self, &problem.ic_options);
        let root = drive_segments(&mut solver, run.times, &mut recorder, |solver, t_next| {
            restarter.restart_sens(solver, t_next)
        })?;
        statistics.solver = solver.get_statistics().clone();
        Ok(recorder.finish(root, &solver.state(), statistics))
    }
}

/// What one solve integrates over, and the workspace its callbacks share.
#[derive(Clone, Copy)]
struct Run<'r> {
    times: TimePlan<'r>,
    ws: &'r RefCell<Workspace>,
    inputs: &'r [f64],
    /// Report output rows rather than states.
    with_output: bool,
    /// When the solve started, for its integration time.
    started: Instant,
}

/// Reads each output column off the solver and files its rows and sensitivities.
struct Recorder<'p> {
    prepared: &'p PreparedSolver,
    ws: &'p RefCell<Workspace>,
    inputs: &'p [f64],
    started: Instant,
    /// The column being read: the state, and `dy` when the trajectory stores `yp`.
    y: FaerVec<f64>,
    dy: Option<FaerVec<f64>>,
    /// `(t, dy)` made consistent for the column at `t` that starts a segment.
    held_dy: Option<(f64, Vec<f64>)>,
    hold_scratch: HoldScratch,
    rows: Rows,
    sens: Option<SensColumns>,
}

/// Where a solve files its rows.
enum Rows {
    /// Straight into a state trajectory.
    States(DenseTrajectory),
    /// Staged for batched evaluation into an outputs trajectory.
    Outputs(OutputBatchWindow),
}

/// The sensitivity half of a solve's columns.
struct SensColumns {
    /// `S` as read for the current column, one lane per parameter.
    lanes: FaerVec<f64>,
    /// One block per parameter in the trajectory's layout.
    blocks: Vec<Vec<f64>>,
    /// `S` at the start, one `n_states` block per parameter.
    start: Vec<Vec<f64>>,
    /// Projects `S` onto the output rows, on an outputs request.
    outputs: Option<OutputSensBuffers>,
}

impl SensColumns {
    /// File the sensitivities of the column at `(t, y)`, read into `self.lanes`.
    fn push(
        &mut self,
        compiled: &CompiledModel,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
    ) {
        if let Some(outputs) = &mut self.outputs {
            outputs.project(compiled, ws, t, y, inputs, &self.lanes);
            for (block, column) in self.blocks.iter_mut().zip(outputs.sens_columns()) {
                block.extend_from_slice(column);
            }
        } else {
            for (block, lane) in self.blocks.iter_mut().zip(lanes(&self.lanes)) {
                block.extend_from_slice(lane);
            }
        }
    }
}

impl<'p> Recorder<'p> {
    /// A recorder for `run`, whose solver starts at `start`.
    fn new(
        prepared: &'p PreparedSolver,
        run: Run<'p>,
        sensitivities: bool,
        start: &StateRef<'_, FaerVec<f64>>,
    ) -> Self {
        let n_points = run.times.eval.len();
        let trajectory = prepared.empty_trajectory(n_points, run.with_output, lane(start.y));
        let n_columns = prepared.sens_binding.len();
        let sens = sensitivities.then(|| SensColumns {
            lanes: start.s.clone(),
            // One reservation per block: `vec![v; n]` would clone away all but one.
            blocks: (0..n_columns)
                .map(|_| Vec::with_capacity(trajectory.n_rows * n_points))
                .collect(),
            start: sens_blocks(start.s),
            outputs: run
                .with_output
                .then(|| OutputSensBuffers::new(prepared, n_columns)),
        });
        let dy = trajectory.stores_yp().then(|| start.dy.clone());
        let rows = if run.with_output {
            Rows::Outputs(OutputBatchWindow::new(prepared.n_states(), trajectory))
        } else {
            Rows::States(trajectory)
        };
        Self {
            prepared,
            ws: run.ws,
            inputs: run.inputs,
            started: run.started,
            y: start.y.clone(),
            dy,
            held_dy: None,
            hold_scratch: HoldScratch::default(),
            rows,
            sens,
        }
    }

    /// Hold the consistent derivative for a segment starting at `t`, if the solver sits on it.
    fn begin(&mut self, state: &StateRef<'_, FaerVec<f64>>, t: f64) {
        self.held_dy = if self.dy.is_some() && within_round_off(state.t, t) {
            let mut ws = self.ws.borrow_mut();
            let held =
                self.prepared
                    .consistent_dy(&mut ws, self.inputs, state, t, &mut self.hold_scratch);
            held.map(|dy| (t, dy))
        } else {
            None
        };
    }

    /// Read one column off `solver` and file it.
    fn record<'a, S: OdeSolverMethod<'a, Equations>>(
        &mut self,
        solver: &S,
        column: ColumnSource,
    ) -> Result<(), CoreError> {
        let sens = self.sens.as_mut().map(|sens| &mut sens.lanes);
        let t = match column {
            ColumnSource::Interpolated(t) => {
                solver.interpolate_inplace(t, &mut self.y)?;
                if let Some(s) = sens {
                    solver.interpolate_sens_inplace(t, s)?;
                }
                if let Some(dy) = self.dy.as_mut() {
                    solver.interpolate_dy_inplace(t, dy)?;
                }
                t
            },
            // The state sits at, or within round-off of, this column, past interpolation's reach.
            ColumnSource::CurrentState(t) => {
                let state = solver.state();
                self.y.copy_from(state.y);
                if let Some(s) = sens {
                    s.copy_from(state.s);
                }
                if let Some(dy) = self.dy.as_mut() {
                    dy.copy_from(state.dy);
                }
                t
            },
        };
        #[allow(clippy::float_cmp)] // the column is read at exactly the held time
        if let (Some(dy), Some((_, held))) = (
            self.dy.as_mut(),
            self.held_dy.take_if(|(t_held, _)| *t_held == t),
        ) {
            lane_mut(dy).copy_from_slice(&held);
        }

        let (compiled, y) = (&self.prepared.compiled, lane(&self.y));
        match &mut self.rows {
            Rows::States(trajectory) => trajectory.push_column(t, y, self.dy.as_ref().map(lane)),
            Rows::Outputs(window) => window.push(t, y, compiled, self.ws, self.inputs),
        }
        if let Some(sens) = &mut self.sens {
            sens.push(compiled, &mut self.ws.borrow_mut(), t, y, self.inputs);
        }
        Ok(())
    }

    /// Package every filed column as the outcome of a solve ended by `root`, if any.
    ///
    /// `end` is the solver's final state, wound back to the root on an event.
    fn finish(
        self,
        root: Option<f64>,
        end: &StateRef<'_, FaerVec<f64>>,
        mut statistics: SolverStatistics,
    ) -> SolveOutcome {
        let rows_are_outputs = matches!(self.rows, Rows::Outputs(_));
        let trajectory = match self.rows {
            Rows::States(trajectory) => trajectory,
            Rows::Outputs(window) => window.finish(
                &self.prepared.compiled,
                &mut self.ws.borrow_mut(),
                self.inputs,
            ),
        };
        statistics.integration_time_secs = self.started.elapsed().as_secs_f64();
        let mut outcome = trajectory.into_outcome(root, lane(end.y), rows_are_outputs, statistics);
        if let Some(sens) = self.sens {
            // Read where y_event was, so the two describe the same point.
            outcome.y_event_sens = outcome.y_event.is_some().then(|| sens_blocks(end.s));
            outcome.sensitivities = Some(sens.blocks);
            outcome.y_start_sens = Some(sens.start);
        }
        outcome
    }
}

/// Interpolation points staged before one batched output evaluation.
const OUTPUT_BATCH_LANES: usize = 128;

/// Staging buffer for the batched output-variable path, filling an outputs trajectory.
struct OutputBatchWindow {
    staged_t: Vec<f64>,
    /// `(n_states, k)` F-contiguous staged states.
    staged_y: Vec<f64>,
    /// `(n_rows, k)` F-contiguous per-window results.
    window_out: Vec<f64>,
    trajectory: DenseTrajectory,
}

impl OutputBatchWindow {
    fn new(n_states: usize, trajectory: DenseTrajectory) -> Self {
        Self {
            staged_t: Vec::with_capacity(OUTPUT_BATCH_LANES),
            staged_y: Vec::with_capacity(n_states * OUTPUT_BATCH_LANES),
            window_out: vec![0.0; trajectory.n_rows * OUTPUT_BATCH_LANES],
            trajectory,
        }
    }

    /// Stage the state `y`, all `n_states` of it, at `t`; evaluates the window once it is full.
    fn push(
        &mut self,
        t: f64,
        y: &[f64],
        compiled: &CompiledModel,
        ws: &RefCell<Workspace>,
        inputs: &[f64],
    ) {
        self.staged_t.push(t);
        self.staged_y.extend_from_slice(y);
        if self.staged_t.len() == OUTPUT_BATCH_LANES {
            self.flush(compiled, &mut ws.borrow_mut(), inputs);
        }
    }

    fn flush(&mut self, compiled: &CompiledModel, ws: &mut Workspace, inputs: &[f64]) {
        let k = self.staged_t.len();
        if k == 0 {
            return;
        }
        let window_out = &mut self.window_out[..self.trajectory.n_rows * k];
        compiled.eval_outputs_batch(ws, k, &self.staged_t, &self.staged_y, inputs, window_out);
        self.trajectory.push_columns(&self.staged_t, window_out);
        self.staged_t.clear();
        self.staged_y.clear();
    }

    /// Flush the last, partial window and hand back the filled trajectory.
    fn finish(
        mut self,
        compiled: &CompiledModel,
        ws: &mut Workspace,
        inputs: &[f64],
    ) -> DenseTrajectory {
        self.flush(compiled, ws, inputs);
        self.trajectory
    }
}

/// Buffers projecting state sensitivities onto the outputs: `dg/dp + dg/dy · S`.
struct OutputSensBuffers {
    n_outputs: usize,
    /// `S` gathered from its lanes, `n_states x k` column-major.
    state_sens: Vec<f64>,
    /// `dg/dp + dg/dy · S`, `n_outputs x k` column-major.
    output_sens: Vec<f64>,
}

impl OutputSensBuffers {
    fn new(prepared: &PreparedSolver, n_columns: usize) -> Self {
        let n_outputs = prepared.n_outputs();
        Self {
            n_outputs,
            state_sens: vec![0.0; prepared.n_states() * n_columns],
            output_sens: vec![0.0; n_outputs * n_columns],
        }
    }

    /// Project `S`, held in `sens_lanes`, onto the outputs at `(t, y)`.
    fn project(
        &mut self,
        compiled: &CompiledModel,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_lanes: &FaerVec<f64>,
    ) {
        for (block, lane) in self
            .state_sens
            .chunks_exact_mut(y.len())
            .zip(lanes(sens_lanes))
        {
            block.copy_from_slice(lane);
        }
        compiled.output_sens_project(ws, t, y, inputs, &self.state_sens, &mut self.output_sens);
    }

    /// The sensitivity columns from the last [`Self::project`], one per parameter.
    fn sens_columns(&self) -> std::slice::ChunksExact<'_, f64> {
        self.output_sens.chunks_exact(self.n_outputs)
    }
}

/// Reject what IDAS rejects: a non-finite or negative tolerance, or a zero `rtol` and `atol` pair.
///
/// diffsol takes any value, and a NaN weight stalls the step controller rather than failing.
fn validate_tolerances(rtol: f64, atol: &[f64]) -> Result<(), CoreError> {
    let invalid = |value: f64| !value.is_finite() || value < 0.0;
    if invalid(rtol) {
        return Err(CoreError::Tolerance {
            name: "rtol",
            index: None,
            got: rtol,
        });
    }
    let bad_atol = atol
        .iter()
        .position(|&value| invalid(value) || (value == 0.0 && rtol == 0.0));
    bad_atol.map_or(Ok(()), |index| {
        Err(CoreError::Tolerance {
            name: "atol",
            index: Some(index),
            got: atol[index],
        })
    })
}

/// How a pure ODE's `dy/dt` follows from `M dy/dt = f`.
enum OdeMass {
    /// `M` has nothing nonzero off its diagonal.
    Diagonal(Vec<f64>),
    /// `M`'s LU, for an `M` with off-diagonal entries.
    General(Box<Lu<usize, f64>>),
}

impl OdeMass {
    /// `None` for a non-diagonal `M` that cannot be factored.
    fn new(mass: &CsrData, sparsity: &FaerSparsity, values: &[f64]) -> Option<Self> {
        mass_diagonal(mass).map(Self::Diagonal).or_else(|| {
            let mass = SparseColMatRef::new(sparsity.as_ref(), values);
            mass.sp_lu().ok().map(|lu| Self::General(Box::new(lu)))
        })
    }

    fn diagonal(&self) -> Option<&[f64]> {
        match self {
            Self::Diagonal(diagonal) => Some(diagonal),
            Self::General(_) => None,
        }
    }

    /// Overwrite `f` with `M⁻¹ f`.
    fn solve_in_place(&self, f: &mut [f64]) {
        match self {
            Self::Diagonal(diagonal) => {
                for (f, m) in f.iter_mut().zip(diagonal) {
                    *f /= m;
                }
            },
            Self::General(lu) => {
                let n = f.len();
                lu.solve_in_place(MatMut::from_column_major_slice_mut(f, n, 1));
            },
        }
    }
}

/// M's diagonal, or `None` if any stored nonzero sits off it.
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

/// The Jacobian's algebraic block `g_v`, laid out once for every held derivative.
struct AlgebraicBlock {
    /// The algebraic states, ascending: `g_v`'s rows and columns in order.
    states: Vec<usize>,
    /// Where each `g_v` entry sits in the Jacobian's CSC values, in `g_v`'s CSC order.
    slots: Vec<usize>,
    pattern: SymbolicSparseColMat<usize>,
    /// `None` when `g_v` is structurally singular.
    symbolic_lu: Option<SymbolicLu<usize>>,
}

impl AlgebraicBlock {
    fn new(csc: &CscPattern, algebraic: &[bool]) -> Self {
        let states: Vec<usize> = (0..algebraic.len()).filter(|&i| algebraic[i]).collect();
        let mut local = vec![usize::MAX; algebraic.len()];
        for (k, &i) in states.iter().enumerate() {
            local[i] = k;
        }
        let (mut col_ptr, mut row_idx, mut slots) = (vec![0], Vec::new(), Vec::new());
        for &col in &states {
            for slot in csc.colptr()[col]..csc.colptr()[col + 1] {
                let row = csc.rowind()[slot];
                if algebraic[row] {
                    row_idx.push(local[row]);
                    slots.push(slot);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let n = states.len();
        let pattern = SymbolicSparseColMat::new_checked(n, n, col_ptr, None, row_idx);
        let symbolic_lu = SymbolicLu::try_new(pattern.as_ref()).ok();
        Self {
            states,
            slots,
            pattern,
            symbolic_lu,
        }
    }

    /// Factor `g_v` out of the full Jacobian's CSC `values`, or `None` if it is singular.
    fn factor(&self, values: &[f64]) -> Option<Lu<usize, f64>> {
        let symbolic = self.symbolic_lu.clone()?;
        let g_v: Vec<f64> = self.slots.iter().map(|&slot| values[slot]).collect();
        Lu::try_new_with_symbolic(symbolic, SparseColMatRef::new(self.pattern.as_ref(), &g_v)).ok()
    }
}

/// The Newton solve behind a consistent state, line-searched like diffsol's default.
type IcNewton =
    NewtonNonlinearSolver<FaerSparseMat<f64>, SparseLu, BacktrackingLineSearch<FaerVec<f64>>>;

/// An [`IcNewton`] configured as diffsol's `new_and_consistent` configures its own.
///
/// Always line-searched: no solve here clears `use_linesearch`.
fn ic_newton(options: &InitialConditionSolverOptions<f64>) -> IcNewton {
    let mut line_search = BacktrackingLineSearch::default();
    line_search.c = options.armijo_constant;
    line_search.max_iter = options.max_linesearch_iterations;
    line_search.tau = options.step_reduction_factor;
    NewtonNonlinearSolver::new(SparseLu::default(), line_search)
}

/// Newton solvers and buffers reused by every breakpoint restart in one solve.
///
/// Only the state solve is line-searched; the sensitivity system is linear, so its
/// undamped step is already exact.
struct BreakpointRestarter<'p> {
    root_solver: IcNewton,
    sens_solver: NewtonNonlinearSolver<FaerSparseMat<f64>, SparseLu, NoLineSearch>,
    /// M's diagonal for a pure ODE with diagonal M; `None` leaves `dy` to `set_consistent`.
    ode_mass_diagonal: Option<&'p [f64]>,
}

impl<'p> BreakpointRestarter<'p> {
    fn new(prepared: &'p PreparedSolver, ic_options: &InitialConditionSolverOptions<f64>) -> Self {
        Self {
            root_solver: ic_newton(ic_options),
            sens_solver: NewtonNonlinearSolver::new(SparseLu::default(), NoLineSearch),
            ode_mass_diagonal: prepared.ode_mass.as_ref().and_then(OdeMass::diagonal),
        }
    }

    /// Rebuild `dy` from `M dy = f(t, y)` when nothing else will.
    ///
    /// diffsol's `set_consistent` skips an ODE, leaving `dy` at the pre-discontinuity slope.
    fn refresh_ode_dy(
        &self,
        problem: &OdeSolverProblem<Equations>,
        state: &mut StateRefMut<'_, FaerVec<f64>>,
    ) {
        let Some(diagonal) = self.ode_mass_diagonal else {
            return;
        };
        problem.eqn.rhs().call_inplace(state.y, *state.t, state.dy);
        for (dy, m) in lane_mut(state.dy).iter_mut().zip(diagonal) {
            *dy /= m;
        }
    }

    /// Move onto the far side of the corner at `t_next` and make `y` and `dy` consistent.
    fn make_consistent<'a, S: OdeSolverMethod<'a, Equations>>(
        &mut self,
        solver: &mut S,
        t_next: f64,
    ) -> Result<(), CoreError> {
        let problem = solver.problem();
        let mut state = solver.state_mut();
        cross_corner(&mut state, t_next);
        state.set_consistent(problem, &mut self.root_solver)?;
        self.refresh_ode_dy(problem, &mut state);
        Ok(())
    }

    /// Restart the integrator at a breakpoint, as IDAKLU's `HandleBreakpoint` does.
    fn restart<'a, S: OdeSolverMethod<'a, Equations>>(
        &mut self,
        solver: &mut S,
        t_next: f64,
    ) -> Result<(), CoreError> {
        self.make_consistent(solver, t_next)?;
        reset_step_size(solver);
        Ok(())
    }

    /// [`Self::restart`] for a solve carrying forward sensitivities.
    fn restart_sens<'a, S>(&mut self, solver: &mut S, t_next: f64) -> Result<(), CoreError>
    where
        S: AugmentedOdeSolverMethod<'a, Equations, SensEquations<'a, Equations>>,
    {
        self.make_consistent(solver, t_next)?;
        // Reads the y and dy just made consistent, so it has to follow them.
        let problem = solver.problem();
        if let Some((mut state, augmented_eqn)) = solver.state_and_augmented_eqn_mut() {
            state.set_consistent_augmented(problem, augmented_eqn, &mut self.sens_solver)?;
        }
        reset_step_size(solver);
        Ok(())
    }
}

/// Pick a fresh first-order step size, as diffsol does after its own IC solve.
fn reset_step_size<'a, S: OdeSolverMethod<'a, Equations>>(solver: &mut S) {
    let problem = solver.problem();
    solver
        .state_mut()
        .set_step_size(problem.h0, &problem.atol, problem.rtol, &problem.eqn, 1);
}

/// Buffers reused by every consistent derivative a solve holds.
#[derive(Default)]
struct HoldScratch {
    /// The Jacobian's CSC values.
    jacobian: Vec<f64>,
    /// The `dF/dt` tape's scratch.
    time_scratch: Vec<f64>,
}

impl PreparedSolver {
    /// `dy` at `state` made consistent, or `None` to keep the interpolated slope.
    ///
    /// `None` when `g_v`, or a pure ODE's non-diagonal `M`, cannot be factored.
    fn consistent_dy(
        &self,
        ws: &mut Workspace,
        inputs: &[f64],
        state: &StateRef<'_, FaerVec<f64>>,
        t: f64,
        scratch: &mut HoldScratch,
    ) -> Option<Vec<f64>> {
        let compiled = &self.compiled;
        let algebraic = self.algebraic_states();
        let (y, mut dy) = (lane(state.y), lane(state.dy).to_vec());
        if !algebraic.contains(&true) {
            // diffsol seeds an ODE's dy with f itself, never solving M dy = f.
            let ode_mass = self.ode_mass.as_ref()?;
            compiled.eval_rhs(ws, t, y, inputs, &mut dy);
            ode_mass.solve_in_place(&mut dy);
            return Some(dy);
        }

        let block = self.algebraic_block.as_ref()?;
        let n = self.n_states();
        let du: Vec<f64> = dy
            .iter()
            .zip(algebraic)
            .map(|(&d, &is_algebraic)| if is_algebraic { 0.0 } else { d })
            .collect();
        let mut g_u_du = vec![0.0; n];
        compiled.jac_action(ws, t, y, inputs, &du, &mut g_u_du);
        let mut time_derivative = vec![0.0; n];
        if let Some(expr) = &self.time_derivative {
            scratch.time_scratch.clear();
            scratch.time_scratch.resize(expr.scratch_len(), 0.0);
            let result = expr.eval(&mut scratch.time_scratch, t, y, &[], inputs);
            time_derivative[..result.len()].copy_from_slice(result);
        }

        scratch.jacobian.clear();
        scratch.jacobian.resize(compiled.csc_sparsity().nnz(), 0.0);
        compiled.assemble_jacobian_csc_no_mass(ws, t, y, inputs, &mut scratch.jacobian);
        let mut dv = Mat::from_fn(block.states.len(), 1, |k, _| {
            let i = block.states[k];
            -(g_u_du[i] + time_derivative[i])
        });
        block.factor(&scratch.jacobian)?.solve_in_place(dv.as_mut());
        for (k, &i) in block.states.iter().enumerate() {
            dy[i] = dv[(k, 0)];
        }
        dy.iter().all(|d| d.is_finite()).then_some(dy)
    }
}

/// The times a solve reports at, together with the stop times inside them.
///
/// `eval` is non-empty and never decreases: [`drive_segments`] would otherwise report a
/// truncated solve.
#[derive(Clone, Copy, Debug)]
struct TimePlan<'a> {
    eval: &'a [f64],
    stop: &'a [f64],
}

impl<'a> TimePlan<'a> {
    /// Check `eval`; equal times are discontinuity brackets, so only a decrease or a
    /// non-finite time fails.
    const fn new(eval: &'a [f64], stop: &'a [f64]) -> Result<Self, CoreError> {
        if eval.is_empty() {
            return Err(CoreError::EmptyTimePoints);
        }
        let mut i = 0;
        while i < eval.len() {
            let got = eval[i];
            if !got.is_finite() {
                return Err(CoreError::NonFiniteTimePoint { index: i, got });
            }
            if i > 0 && got < eval[i - 1] {
                return Err(CoreError::UnsortedTimePoints {
                    index: i,
                    got,
                    previous: eval[i - 1],
                });
            }
            i += 1;
        }
        Ok(Self { eval, stop })
    }

    /// The first output time, where integration starts.
    const fn t0(&self) -> f64 {
        self.eval[0]
    }

    fn segments(&self) -> Vec<Range<usize>> {
        segment_ranges(self.eval, self.stop)
    }
}

/// Where one output column's values come from.
///
/// Interpolated once the solver has stepped on or past it; read off the state only for an
/// event's root and for columns left within round-off past where the state stopped.
#[derive(Clone, Copy, Debug)]
enum ColumnSource {
    Interpolated(f64),
    CurrentState(f64),
}

/// Step one `t_stop` segment at a time, recording every output column.
///
/// Returns the root time if an event ended the solve, with the solver wound back to it.
fn drive_segments<'a, S: OdeSolverMethod<'a, Equations>>(
    solver: &mut S,
    times: TimePlan<'_>,
    recorder: &mut Recorder<'_>,
    mut restart: impl FnMut(&mut S, f64) -> Result<(), CoreError>,
) -> Result<Option<f64>, CoreError> {
    let t_eval = times.eval;
    let mut col = 0;
    for (i, range) in times.segments().into_iter().enumerate() {
        let last = range.end - 1;
        if i > 0 {
            // The far side of the ULP-bracketed corner, where the new branch is evaluable.
            restart(solver, t_eval[range.start])?;
        }
        // A consistent-state solve can jump past a root the root finder would then miss.
        if event_non_positive(solver) {
            let t = solver.state().t;
            recorder.record(solver, ColumnSource::CurrentState(t))?;
            return Ok(Some(t));
        }
        recorder.begin(&solver.state(), t_eval[range.start]);
        // `set_stop_time` rejects a stop diffsol already counts as reached.
        if !stop_time_reached(&solver.state(), t_eval[last]) {
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
                    recorder.record(solver, ColumnSource::Interpolated(t_eval[col]))?;
                    col += 1;
                }
                match reason {
                    OdeSolverStopReason::InternalTimestep => {},
                    OdeSolverStopReason::TstopReached => break,
                    OdeSolverStopReason::RootFound(t_root, _) => {
                        solver.state_mut_back(t_root)?;
                        if col <= last {
                            recorder.record(solver, ColumnSource::CurrentState(t_root))?;
                        }
                        return Ok(Some(t_root));
                    },
                }
            }
        }
        // Whatever is left lies within diffsol's stop round-off, past where the state stopped.
        for &t in &t_eval[col..=last] {
            recorder.record(solver, ColumnSource::CurrentState(t))?;
        }
        col = last + 1;
    }
    Ok(None)
}

/// diffsol's own test for a stop at `t_stop` being reached, round-off included.
fn stop_time_reached(state: &StateRef<'_, FaerVec<f64>>, t_stop: f64) -> bool {
    (state.t - t_stop).abs() <= 100.0 * f64::EPSILON * (state.t.abs() + state.h.abs())
}

/// Whether any termination event is non-positive at the solver's current state.
fn event_non_positive<'a, S: OdeSolverMethod<'a, Equations>>(solver: &S) -> bool {
    let Some(events) = solver.problem().eqn.root() else {
        return false;
    };
    let state = solver.state();
    let mut values = FaerVec::zeros(events.nout(), *state.y.context());
    events.call_inplace(state.y, state.t, &mut values);
    lane(&values).iter().any(|&value| value <= 0.0)
}

/// Move a restart at a corner onto its far side, `t_next`, where the new branch holds.
fn cross_corner(state: &mut StateRefMut<'_, FaerVec<f64>>, t_next: f64) {
    if within_round_off(*state.t, t_next) {
        *state.t = t_next;
    }
}

/// Whether `from` and `to` are one instant to diffsol's stop round-off, without its step term.
///
/// `PyBaMM`'s ULP-apart discontinuity brackets collapse to the earlier, pre-discontinuity time.
fn within_round_off(from: f64, to: f64) -> bool {
    (to - from).abs() <= 100.0 * f64::EPSILON * from.abs().max(to.abs())
}

/// One range of `t_eval` per stretch between consecutive stop times.
///
/// A stop time not on the output grid is integrated through rather than ending a segment.
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
        // Keeps cuts increasing, so an unsorted `t_stop` cannot cause a slice panic.
        let previous = cuts.last().copied().unwrap_or(0);
        if idx <= previous {
            continue;
        }
        if within_round_off(t_eval[previous], t_eval[idx]) {
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
const MAX_HERMITE_COLUMNS: usize = 4096;

/// A column-major trajectory accumulated one output column at a time.
struct DenseTrajectory {
    t: Vec<f64>,
    y: Vec<f64>,
    /// `dy/dt` columns sharing `y`'s layout, when stored.
    yp: Option<Vec<f64>>,
    /// Trajectory width: states, or outputs on the output paths.
    n_rows: usize,
    y_start: Vec<f64>,
}

impl DenseTrajectory {
    fn with_capacity(n_rows: usize, n_points: usize, store_yp: bool) -> Self {
        Self {
            t: Vec::with_capacity(n_points),
            y: Vec::with_capacity(n_rows * n_points),
            yp: store_yp.then(|| Vec::with_capacity(n_rows * n_points)),
            n_rows,
            y_start: Vec::new(),
        }
    }

    /// Package the trajectory as a solve's outcome, ended by `root` if an event fired.
    ///
    /// `final_state` is the full state where the trajectory ends. It is reported at a root,
    /// and always when the rows are outputs, as then it is the caller's only state.
    fn into_outcome(
        self,
        root: Option<f64>,
        final_state: &[f64],
        rows_are_outputs: bool,
        statistics: SolverStatistics,
    ) -> SolveOutcome {
        SolveOutcome {
            n_rows: self.n_rows,
            t: self.t,
            y: self.y,
            yp: self.yp,
            sensitivities: None,
            y_event: (root.is_some() || rows_are_outputs).then(|| final_state.to_vec()),
            termination: root.map_or(Termination::Completed, |t| Termination::Event { t }),
            y_start: self.y_start,
            y_start_sens: None,
            y_event_sens: None,
            statistics,
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

    /// Append one column per entry of `t`, stored back to back in `y`, storing no `yp`.
    fn push_columns(&mut self, t: &[f64], y: &[f64]) {
        debug_assert_eq!(
            y.len(),
            t.len() * self.n_rows,
            "column block width mismatch"
        );
        debug_assert!(self.yp.is_none(), "a column block carries no yp");
        self.t.extend_from_slice(t);
        self.y.extend_from_slice(y);
    }

    const fn stores_yp(&self) -> bool {
        self.yp.is_some()
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
        // diffsol rejects the second of a ULP-apart pair as a stop already reached.
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
    fn a_bracket_at_the_end_of_the_span_still_cuts() {
        // The final point restarts on the far side of the corner, as a mid-span bracket does.
        let below = f64::from_bits(3.0f64.to_bits() - 1);
        let t = [0.0, 1.0, below, 3.0];
        assert_eq!(segment_ranges(&t, &[below]), vec![0..3, 3..4]);
    }
}
#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use diffsol::{Matrix, NonLinearOpSens, OdeEquations, Op, Vector};
    use faer::sparse::{SparseColMat, Triplet};

    use super::*;
    use crate::arena::Arena;
    use crate::model::test_fixtures::{compile, compile_with_sens, identity_mass_matrix, input};
    use crate::node::{CsrData, Node, Shape};
    use crate::observable::ObservableKind;
    use crate::params::ParamIdx;
    use crate::solver::test_fixtures::{
        decay, decay_with_output_and_event, differential_then_algebraic_mass,
    };

    /// `dy/dt = -a * y` and `0 = y - z`, with sensitivities wrt `a`.
    fn build_small_dae_with_param() -> CompiledModel {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg_a_y = {
            let neg = arena.alloc(Node::Scalar(-1.0));
            let neg_a = arena.alloc(Node::Mul(neg, a));
            arena.alloc(Node::Mul(neg_a, sv0))
        };
        let algebraic_residual = arena.alloc(Node::Sub(sv0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![neg_a_y, algebraic_residual]));
        let mass = differential_then_algebraic_mass();
        compile_with_sens(&arena, rhs, mass, 2, 1, &[ParamIdx::new(0)])
    }

    /// A Heaviside node constructor, such as `Node::EqualHeaviside`.
    type Switch = fn(NodeId, NodeId) -> Node;

    /// `dv/dt = -a*v` with `0 = z - v*(1 - 2*H)`, `H = switch(0.5, t)`: `z` jumps from `v` to `-v`.
    fn build_dae_with_algebraic_jump(switch: Switch, terminate: bool) -> CompiledModel {
        let mut arena = Arena::new();
        let v = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let dvdt = arena.alloc(Node::Mul(neg_a, v));
        let t = arena.alloc(Node::Time);
        let corner = arena.alloc(Node::Scalar(0.5));
        let after = arena.alloc(switch(corner, t));
        let two = arena.alloc(Node::Scalar(2.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let two_after = arena.alloc(Node::Mul(two, after));
        let sign = arena.alloc(Node::Sub(one, two_after));
        let target = arena.alloc(Node::Mul(v, sign));
        let constraint = arena.alloc(Node::Sub(z, target));
        let rhs = arena.alloc(Node::Concat(vec![dvdt, constraint]));
        let mass = differential_then_algebraic_mass();
        let mut model = compile_with_sens(&arena, rhs, mass, 2, 1, &[ParamIdx::new(0)]);
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[z])
            .expect("a valid output");
        if terminate {
            model
                .set_observables(ObservableKind::Events, &mut arena, &[z])
                .expect("a valid event");
        }
        model
    }

    #[test]
    fn a_corner_restart_lands_the_algebraic_state_on_the_new_branch() {
        // PyBaMM brackets `t > 0.5` with 0.5 and the time one ULP above it.
        let above = f64::from_bits(0.5f64.to_bits() + 1);
        let t_eval = [0.0, 0.25, 0.5, above, 0.75, 1.0, 1.5, 2.0];
        let a = 0.1;
        let request = SolveRequest::new(&t_eval).with_stop_times(&t_eval);
        for (store_yp, request) in [
            (false, request),
            (true, request),
            (false, request.with_sensitivities()),
        ] {
            let prepared = PreparedSolver::new(
                build_dae_with_algebraic_jump(Node::NotEqualHeaviside, false),
                1e-6,
                &[1e-6, 1e-6],
            )
            .expect("prepare")
            .with_store_yp(store_yp);
            let result = prepared
                .solve(request, InputSet::new(&[1.0, 1.0], &[a]))
                .expect("a jump in an algebraic state at a corner must not stall the solve");

            for (j, &t) in t_eval.iter().enumerate() {
                let sign = if t > 0.5 { -1.0 } else { 1.0 };
                let v = (-a * t).exp();
                let got = &result.y[j * result.n_rows..(j + 1) * result.n_rows];
                assert!(
                    (got[0] - v).abs() < 1e-5,
                    "v({t}) = {} but expected {v}",
                    got[0]
                );
                assert!((got[1] - sign * v).abs() < 1e-5, "z({t}) = {}", got[1]);
                if let Some(sens) = &result.sensitivities {
                    let dv_da = -t * v;
                    let got = &sens[0][j * result.n_rows..(j + 1) * result.n_rows];
                    assert!((got[0] - dv_da).abs() < 1e-5, "dv/da({t}) = {}", got[0]);
                    assert!(
                        (got[1] - sign * dv_da).abs() < 1e-5,
                        "dz/da({t}) = {}",
                        got[1]
                    );
                }
            }
        }
    }

    #[test]
    fn a_corner_restart_reports_a_termination_event_crossed_by_the_jump() {
        let above = f64::from_bits(0.5f64.to_bits() + 1);
        let times = [0.0, 0.25, 0.5, above, 0.75, 1.0];
        let request = SolveRequest::new(&times).with_stop_times(&times);
        let prepared = PreparedSolver::new(
            build_dae_with_algebraic_jump(Node::NotEqualHeaviside, true),
            1e-8,
            &[1e-8, 1e-8],
        )
        .expect("prepare");

        for request in [
            request,
            request.with_outputs(),
            request.with_sensitivities(),
            request.with_outputs().with_sensitivities(),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&[1.0, 1.0], &[0.1]))
                .expect("solve");
            assert_eq!(result.termination, Termination::Event { t: above });
            assert_eq!(result.t, times[..4]);
            let event_state = result.y_event.expect("event state");
            assert!((event_state[0] - (-0.05_f64).exp()).abs() < 1e-6);
            assert!((event_state[1] + (-0.05_f64).exp()).abs() < 1e-6);
        }
    }

    #[test]
    fn a_bracket_at_the_end_of_the_span_restarts_on_the_new_branch() {
        // PyBaMM brackets a switch at t_end as [0.5, above] for `t > 0.5`, [below, 0.5] for `>=`.
        let below = f64::from_bits(0.5f64.to_bits() - 1);
        let above = f64::from_bits(0.5f64.to_bits() + 1);
        let a = 0.1;
        let v = (-a * 0.5f64).exp();
        let brackets: [(Switch, [f64; 2]); 2] = [
            (Node::NotEqualHeaviside, [0.5, above]),
            (Node::EqualHeaviside, [below, 0.5]),
        ];
        for (switch, bracket) in brackets {
            let t_eval = [0.0, 0.25, bracket[0], bracket[1]];
            let prepared = PreparedSolver::new(
                build_dae_with_algebraic_jump(switch, false),
                1e-8,
                &[1e-8, 1e-8],
            )
            .expect("prepare");
            let request = SolveRequest::new(&t_eval).with_stop_times(&t_eval);
            for request in [request, request.with_sensitivities()] {
                let result = prepared
                    .solve(request, InputSet::new(&[1.0, 1.0], &[a]))
                    .expect("a switch at the final time must not stall the solve");
                let end = &result.y[3 * result.n_rows..];
                assert!((end[0] - v).abs() < 1e-6, "v = {}, {request:?}", end[0]);
                assert!((end[1] + v).abs() < 1e-6, "z = {}, {request:?}", end[1]);
                if let Some(blocks) = &result.sensitivities {
                    // dz/da = -dv/da = t v on the new branch.
                    let dz_da = blocks[0][3 * result.n_rows + 1];
                    assert!((dz_da - 0.5 * v).abs() < 1e-5, "dz/da = {dz_da}");
                }
            }
        }
    }

    /// `0 = atan(z - v - 3*(t > 0.5))`: a jump too far for an undamped Newton step.
    fn build_dae_with_a_far_algebraic_jump() -> CompiledModel {
        let mut arena = Arena::new();
        let v = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let dvdt = arena.alloc(Node::Mul(neg_a, v));
        let t = arena.alloc(Node::Time);
        let corner = arena.alloc(Node::Scalar(0.5));
        let after = arena.alloc(Node::NotEqualHeaviside(corner, t));
        let three = arena.alloc(Node::Scalar(3.0));
        let jump = arena.alloc(Node::Mul(three, after));
        let target = arena.alloc(Node::Add(v, jump));
        let offset = arena.alloc(Node::Sub(z, target));
        let constraint = arena.alloc(Node::Arctan(offset));
        let rhs = arena.alloc(Node::Concat(vec![dvdt, constraint]));
        compile(&arena, rhs, differential_then_algebraic_mass(), 2, 1)
    }

    #[test]
    fn a_corner_restart_line_searches_a_far_algebraic_jump() {
        let above = f64::from_bits(0.5f64.to_bits() + 1);
        let t_eval = [0.0, 0.25, 0.5, above, 1.0];
        let a = 0.1;
        let prepared =
            PreparedSolver::new(build_dae_with_a_far_algebraic_jump(), 1e-6, &[1e-6, 1e-6])
                .expect("prepare");
        let result = prepared
            .solve(
                SolveRequest::new(&t_eval).with_stop_times(&t_eval),
                InputSet::new(&[1.0, 1.0], &[a]),
            )
            .expect("the restart's Newton must reach a far branch, as diffsol's own IC does");

        let j = t_eval.len() - 1;
        let z = result.y[j * result.n_rows + 1];
        let expected = (-a).exp() + 3.0;
        assert!(
            (z - expected).abs() < 1e-5,
            "z(1) = {z} but expected {expected}"
        );
    }

    /// `dy/dt = -a*y` with `0 = 2y - z - z^3`, so the IC Newton stops short of consistency.
    fn build_nonlinear_dae() -> CompiledModel {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let dydt = arena.alloc(Node::Mul(neg_a, y));
        let two = arena.alloc(Node::Scalar(2.0));
        let two_y = arena.alloc(Node::Mul(two, y));
        let z2 = arena.alloc(Node::Mul(z, z));
        let z3 = arena.alloc(Node::Mul(z2, z));
        let z_plus_z3 = arena.alloc(Node::Add(z, z3));
        let constraint = arena.alloc(Node::Sub(two_y, z_plus_z3));
        let rhs = arena.alloc(Node::Concat(vec![dydt, constraint]));
        compile(&arena, rhs, differential_then_algebraic_mass(), 2, 1)
    }

    #[test]
    fn the_first_column_stores_the_consistent_derivative() {
        // From y = 1 the constraint gives z = 1 and dz/dt = 2 dy/dt / (1 + 3z^2) = -0.5.
        let prepared = PreparedSolver::new(build_nonlinear_dae(), 1e-6, &[1e-6, 1e-6])
            .expect("prepare")
            .with_store_yp(true);
        let t_eval: Vec<f64> = (0..=4).map(|i| f64::from(i) * 0.5).collect();
        let result = prepared
            .solve(
                SolveRequest::new(&t_eval),
                InputSet::new(&[1.0, 3.0], &[1.0]),
            )
            .expect("solve");

        assert!(
            (result.y_start[1] - 1.0).abs() < 1e-5,
            "z(0) = {}",
            result.y_start[1]
        );
        let yp = result.yp.expect("yp requested");
        assert!((yp[0] + 1.0).abs() < 1e-9, "dy/dt(0) = {}", yp[0]);
        assert!((yp[1] + 0.5).abs() < 1e-6, "dz/dt(0) = {}", yp[1]);
    }

    #[test]
    // The same held derivative fills both columns, bit for bit.
    #[allow(clippy::float_cmp)]
    fn a_single_column_solve_stores_the_consistent_derivative_too() {
        let prepared = PreparedSolver::new(build_nonlinear_dae(), 1e-6, &[1e-6, 1e-6])
            .expect("prepare")
            .with_store_yp(true);
        let set = InputSet::new(&[1.0, 3.0], &[1.0]);
        let full = prepared
            .solve(SolveRequest::new(&[0.0, 0.5, 1.0]), set)
            .expect("solve");
        let single = prepared
            .solve(SolveRequest::new(&[0.0]), set)
            .expect("a one-point span must solve");
        assert_eq!(single.y, single.y_start);
        assert_eq!(
            single.yp.expect("yp requested"),
            full.yp.expect("yp requested")[..2]
        );
    }

    /// States `[u0, v0, u1, v1]` with `v` algebraic, coupled through a non-diagonal `g_v`.
    fn build_coupled_dae() -> CompiledModel {
        let mut arena = Arena::new();
        let [u0, v0, u1, v1] = [0, 1, 2, 3].map(|i| {
            arena.alloc(Node::StateVector {
                start: i,
                end: i + 1,
            })
        });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let half = arena.alloc(Node::Scalar(0.5));
        let t = arena.alloc(Node::Time);
        let du0 = arena.alloc(Node::Mul(neg, v1));
        let u1v0 = arena.alloc(Node::Mul(u1, v0));
        let du1 = arena.alloc(Node::Mul(neg, u1v0));
        let hv1 = arena.alloc(Node::Mul(half, v1));
        let g0 = arena.alloc(Node::Add(v0, hv1));
        let g0 = arena.alloc(Node::Sub(g0, u0));
        let sin_t = arena.alloc(Node::Sin(t));
        let g0 = arena.alloc(Node::Sub(g0, sin_t));
        let v0v1 = arena.alloc(Node::Mul(v0, v1));
        let g1 = arena.alloc(Node::Add(v1, v0v1));
        let g1 = arena.alloc(Node::Sub(g1, u1));
        let rhs = arena.alloc(Node::Concat(vec![du0, g0, du1, g1]));
        let mass = CsrData {
            indptr: vec![0, 1, 1, 2, 2],
            indices: vec![0, 2],
            data: vec![1.0, 1.0],
            shape: Shape::matrix(4, 4),
        };
        compile(&arena, rhs, mass, 4, 0)
    }

    #[test]
    // Bit-identical is the property under test.
    #[allow(clippy::float_cmp)]
    fn the_cached_algebraic_block_factors_as_a_fresh_triplet_build_does() {
        let prepared = PreparedSolver::new(build_coupled_dae(), 1e-8, &[1e-8; 4])
            .expect("prepare")
            .with_store_yp(true);
        let block = prepared.algebraic_block.as_ref().expect("built for yp");
        let (compiled, csc) = (&prepared.compiled, prepared.compiled.csc_sparsity());
        let algebraic = prepared.algebraic_states();
        let local = |i: usize| {
            block
                .states
                .iter()
                .position(|&s| s == i)
                .expect("algebraic")
        };
        let mut ws = compiled.create_workspace();
        for (t, y) in [(0.3, [1.0, 0.2, 0.5, -0.3]), (2.0, [0.3, -1.1, 2.0, 0.4])] {
            let mut values = vec![0.0; csc.nnz()];
            compiled.assemble_jacobian_csc_no_mass(&mut ws, t, &y, &[], &mut values);
            // The per-hold build this replaced: triplets, a fresh pattern and a full `sp_lu`.
            let mut triplets = Vec::new();
            for &col in &block.states {
                let entries = csc.colptr()[col]..csc.colptr()[col + 1];
                for (&row, &value) in csc.rowind()[entries.clone()].iter().zip(&values[entries]) {
                    if algebraic[row] {
                        triplets.push(Triplet::new(local(row), local(col), value));
                    }
                }
            }
            let n = block.states.len();
            let g_v = SparseColMat::try_new_from_triplets(n, n, &triplets).expect("g_v");
            let rhs = Mat::from_fn(n, 1, |i, _| 1.5 - i as f64);
            let mut expected = rhs.clone();
            g_v.sp_lu().expect("lu").solve_in_place(expected.as_mut());
            let mut got = rhs;
            block
                .factor(&values)
                .expect("lu")
                .solve_in_place(got.as_mut());
            assert_eq!(got, expected, "t = {t}");
        }
    }

    #[test]
    fn a_time_dependent_constraint_takes_its_first_slope_from_the_time_derivative() {
        // dy/dt = -a*y with 0 = z - t: only dF/dt gives dz/dt = 1 at t0.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let dydt = arena.alloc(Node::Mul(neg_a, y));
        let t = arena.alloc(Node::Time);
        let constraint = arena.alloc(Node::Sub(z, t));
        let rhs = arena.alloc(Node::Concat(vec![dydt, constraint]));
        let mass = differential_then_algebraic_mass();
        let model = compile_with_sens(&arena, rhs, mass, 2, 1, &[ParamIdx::new(0)]);
        let prepared = PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8])
            .expect("prepare")
            .with_store_yp(true)
            .with_time_derivative(&arena, rhs);
        let t_eval = [0.0, 0.5, 1.0];

        for request in [
            SolveRequest::new(&t_eval),
            SolveRequest::new(&t_eval).with_sensitivities(),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&[1.0, 0.0], &[1.0]))
                .expect("solve");
            let yp = result.yp.expect("yp requested");
            assert!((yp[1] - 1.0).abs() < 1e-12, "dz/dt(0) = {}", yp[1]);
        }
    }

    #[test]
    fn every_state_path_stores_the_derivative_of_every_column() {
        // With z = y every slope is -a*y on both rows; a stop at every time carries none over.
        let prepared = PreparedSolver::new(build_small_dae_with_param(), 1e-10, &[1e-10, 1e-10])
            .expect("prepare")
            .with_store_yp(true);
        let (y0, inputs, t_eval) = small_dae_setup();
        let plain = SolveRequest::new(&t_eval);
        let stops = plain.with_stop_times(&t_eval);
        for request in [
            plain,
            stops,
            plain.with_sensitivities(),
            stops.with_sensitivities(),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&y0, &inputs))
                .expect("solve");
            let yp = result.yp.expect("yp requested");
            assert_eq!(yp.len(), result.y.len());
            for j in 0..result.t.len() {
                let expected = -inputs[0] * result.y[j * result.n_rows];
                for row in 0..result.n_rows {
                    let got = yp[j * result.n_rows + row];
                    assert!(
                        (got - expected).abs() <= 1e-6 * expected.abs(),
                        "yp[{row}, {j}] = {got} but -a*y = {expected}, {request:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn y_start_is_the_consistent_state_whatever_the_payload() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, t_eval) = small_dae_setup();
        for request in [
            SolveRequest::new(&t_eval),
            SolveRequest::new(&t_eval).with_sensitivities(),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&y0, &inputs))
                .expect("solve");
            assert!((result.y_start[0] - 1.0).abs() < 1e-12);
            assert!(
                (result.y_start[1] - 1.0).abs() < 1e-6,
                "z(0) = {}",
                result.y_start[1]
            );
        }
    }

    /// Like `build_small_dae_with_param` with a second input: `dy/dt = -a*y + b`.
    fn build_small_dae_two_inputs(sens: &[ParamIdx]) -> CompiledModel {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let b = input(&mut arena, "b", 1, 1, 1);
        let ode = {
            let neg = arena.alloc(Node::Scalar(-1.0));
            let neg_a = arena.alloc(Node::Mul(neg, a));
            let neg_a_y = arena.alloc(Node::Mul(neg_a, sv0));
            arena.alloc(Node::Add(neg_a_y, b))
        };
        let algebraic_residual = arena.alloc(Node::Sub(sv0, sv1));
        let rhs = arena.alloc(Node::Concat(vec![ode, algebraic_residual]));
        compile_with_sens(&arena, rhs, differential_then_algebraic_mass(), 2, 2, sens)
    }

    fn build_small_dae_prepared_with_sens() -> PreparedSolver {
        let model = build_small_dae_with_param();
        PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed")
    }

    fn blocks(outcome: &SolveOutcome) -> &[Vec<f64>] {
        outcome
            .sensitivities
            .as_deref()
            .expect("the request asked for sensitivities")
    }

    fn small_dae_setup() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let y0 = vec![1.0, 0.0]; // algebraic IC deliberately inconsistent
        let inputs = vec![1.0];
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();
        (y0, inputs, t_eval)
    }

    /// One state, `dy/dt = -a * y`: no algebraic row, so `set_consistent` returns early.
    fn build_small_ode_with_param() -> CompiledModel {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let rhs = arena.alloc(Node::Mul(neg_a, sv0));
        compile(&arena, rhs, identity_mass_matrix(1), 1, 1)
    }

    /// `M dy/dt = -y` on two states, with the given mass matrix.
    fn build_ode_with_mass(mass: CsrData) -> CompiledModel {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let f0 = arena.alloc(Node::Mul(neg, y0));
        let f1 = arena.alloc(Node::Mul(neg, y1));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1]));
        compile(&arena, rhs, mass, 2, 1)
    }

    #[test]
    fn the_first_column_of_a_pure_ode_divides_out_a_diagonal_mass() {
        // M = diag(2, 4), so dy/dt = (-y0 / 2, -y1 / 4), not f = -y.
        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 4.0],
            shape: Shape::matrix(2, 2),
        };
        let prepared = PreparedSolver::new(build_ode_with_mass(mass), 1e-8, &[1e-8, 1e-8])
            .expect("prepare")
            .with_store_yp(true);
        let t_eval = [0.0, 0.5, 1.0];
        for request in [
            SolveRequest::new(&t_eval),
            SolveRequest::new(&t_eval).with_stop_times(&t_eval),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&[1.0, 2.0], &[0.0]))
                .expect("solve");
            let yp = result.yp.expect("yp requested");
            assert_eq!(yp[..2], [-0.5, -0.5], "{request:?}");
        }
    }

    #[test]
    fn the_first_column_of_a_pure_ode_solves_against_a_non_diagonal_mass() {
        // M = [[2, 1], [0, 1]] gives dy/dt = ((y1 - y0) / 2, -y1), far from f = -y.
        let mass = CsrData {
            indptr: vec![0, 2, 3],
            indices: vec![0, 1, 1],
            data: vec![2.0, 1.0, 1.0],
            shape: Shape::matrix(2, 2),
        };
        let prepared = PreparedSolver::new(build_ode_with_mass(mass), 1e-10, &[1e-10, 1e-10])
            .expect("prepare")
            .with_store_yp(true);
        let result = prepared
            .solve(
                SolveRequest::new(&[0.0, 0.5, 1.0]),
                InputSet::new(&[1.0, 0.5], &[0.0]),
            )
            .expect("solve");
        let yp = result.yp.expect("yp requested");
        assert!((yp[0] + 0.25).abs() < 1e-15, "dy0/dt(0) = {}", yp[0]);
        assert!((yp[1] + 0.5).abs() < 1e-15, "dy1/dt(0) = {}", yp[1]);
    }

    #[test]
    fn a_breakpoint_restart_refreshes_dy_on_a_pure_ode() {
        let prepared =
            PreparedSolver::new(build_small_ode_with_param(), 1e-8, &[1e-8]).expect("prepare");
        let inputs = [2.0_f64];
        let y0 = [1.0_f64];

        let set = InputSet::new(&y0, &inputs);
        let (_ws, problem) = prepared
            .build_problem(set, 0.0, &prepared.packed_binding(), false, Par::Seq)
            .expect("build");
        let state = problem.bdf_state::<SparseLu>().expect("state");
        let mut solver = problem.bdf_solver::<SparseLu>(state).expect("solver");
        // Step away from t0 so `dy` holds a BDF slope rather than its initial seed.
        solver.solve_dense(&[0.0, 0.5, 1.0]).expect("solve");

        let stale = solver.state().dy[0];
        BreakpointRestarter::new(&prepared, &problem.ic_options)
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
    // Every column is the untouched start state.
    #[allow(clippy::float_cmp)]
    fn a_span_that_never_leaves_t0_reports_every_column_at_the_start() {
        let prepared = build_decay_param_output_event().with_store_yp(true);
        for t_eval in [&[0.0][..], &[0.0, 0.0]] {
            let base = SolveRequest::new(t_eval).with_stop_times(t_eval);
            for request in [
                base,
                base.with_outputs(),
                base.with_sensitivities(),
                base.with_outputs().with_sensitivities(),
            ] {
                let result = prepared
                    .solve(request, InputSet::new(&[1.0], &[1.0]))
                    .expect("a span of zero length must not be an error");
                assert_eq!(result.t, t_eval, "{request:?}");
                assert_eq!(result.termination, Termination::Completed);
                let expected = if request.outputs { 2.0 } else { 1.0 };
                assert!(result.y.iter().all(|&y| y == expected), "{request:?}");
                if let Some(blocks) = &result.sensitivities {
                    assert!(blocks[0].iter().all(|&s| s == 0.0), "{request:?}");
                }
            }
        }
    }

    #[test]
    fn output_points_within_round_off_of_a_segment_end_are_all_reported() {
        // One step spans each short span, so diffsol can stop an ULP short of tstop.
        let prepared = build_decay_param_output_event();
        for i in 0..100 {
            let t_end = 1e-6 * (0.2 + 0.0123 * f64::from(i));
            for k in 0..=4 {
                let below = f64::from_bits(t_end.to_bits() - k);
                let t_eval = [0.0, below, t_end];
                for request in [
                    SolveRequest::new(&t_eval),
                    SolveRequest::new(&t_eval).with_sensitivities(),
                ] {
                    let result = prepared
                        .solve(request, InputSet::new(&[1.0], &[1e-3]))
                        .expect("solve");
                    assert_eq!(result.t, t_eval, "t_end = {t_end}, k = {k}");
                }
            }
        }
    }

    #[test]
    fn solve_empty_t_eval_returns_error_not_panic() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        let err = prepared
            .solve(SolveRequest::new(&[]), InputSet::new(&y0, &inputs))
            .expect_err("empty t_eval must be a returned error, not a panic");
        assert!(
            matches!(err, CoreError::EmptyTimePoints),
            "expected EmptyTimePoints, got: {err:?}"
        );
    }

    #[test]
    fn an_unsorted_or_non_finite_t_eval_is_an_error_not_a_wrong_trajectory() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        let rejected = |t_eval: &[f64]| {
            prepared
                .solve(SolveRequest::new(t_eval), InputSet::new(&y0, &inputs))
                .expect_err("a bad t_eval must be an error, not a trajectory")
        };
        for t_eval in [[0.0, 2.0, 1.0], [0.0, 0.5, 0.2]] {
            let err = rejected(&t_eval);
            assert!(
                matches!(err, CoreError::UnsortedTimePoints { index: 2, .. }),
                "{t_eval:?}: {err:?}"
            );
        }
        // -inf once reported every column at the start state; +inf integrated to y = 0.
        for (t_eval, index) in [
            (vec![0.0, f64::NAN, 1.0], 1),
            (vec![f64::NEG_INFINITY, 0.0], 0),
            (vec![0.0, f64::INFINITY], 1),
            (vec![f64::NAN], 0),
        ] {
            let err = rejected(&t_eval);
            assert!(
                matches!(err, CoreError::NonFiniteTimePoint { index: i, .. } if i == index),
                "{t_eval:?}: {err:?}"
            );
        }
    }

    #[test]
    fn output_solves_reject_a_model_without_output_variables() {
        let prepared = build_small_dae_prepared_with_sens(); // no outputs
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
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _t) = small_dae_setup();
        let t_eval = vec![0.0, 0.2, 0.2, 0.4];
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
            .expect("a repeated time must still solve");
        assert_eq!(r.t.len(), t_eval.len());
    }

    #[test]
    fn solve_y0_length_mismatch_returns_error_not_panic() {
        let prepared = build_small_dae_prepared_with_sens(); // n_states = 2
        let (_y0, inputs, t_eval) = small_dae_setup();
        let err = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &inputs))
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
        let prepared = build_small_dae_prepared_with_sens(); // needs 1 packed input
        let (y0, _inputs, t_eval) = small_dae_setup();
        let err = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &[]))
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
    fn tolerances_idas_would_reject_are_rejected_up_front() {
        let good = [1e-8, 1e-8];
        let cases = [
            (f64::NAN, good, "rtol", None),
            (f64::INFINITY, good, "rtol", None),
            (-1e-8, good, "rtol", None),
            (1e-8, [1e-8, f64::NAN], "atol", Some(1)),
            (1e-8, [f64::NEG_INFINITY, 1e-8], "atol", Some(0)),
            (1e-8, [1e-8, -1e-8], "atol", Some(1)),
            (0.0, [1e-8, 0.0], "atol", Some(1)),
        ];
        for (rtol, atol, name, index) in cases {
            let err = PreparedSolver::new(build_small_dae_with_param(), rtol, &atol)
                .expect_err("an invalid tolerance must be rejected");
            assert!(
                matches!(err, CoreError::Tolerance { name: n, index: i, .. } if n == name && i == index),
                "rtol={rtol}, atol={atol:?}: got {err:?}"
            );
        }
        let err = PreparedSolver::new(build_small_dae_with_param(), 0.0, &[1e-8, 0.0])
            .expect_err("rtol and atol both zero");
        assert_eq!(
            err.to_string(),
            "atol[1] = 0 is not a valid tolerance: rtol and atol must be finite and >= 0, and not both zero"
        );
    }

    #[test]
    fn a_zero_rtol_or_a_zero_atol_alone_is_a_valid_tolerance() {
        for (rtol, atol) in [(0.0, [1e-8, 1e-8]), (1e-8, [0.0, 1e-8])] {
            PreparedSolver::new(build_small_dae_with_param(), rtol, &atol)
                .expect("IDAS accepts a zero on one side");
        }
    }

    /// Check sensitivity block 0 against a central difference at the final time.
    ///
    /// `perturbed(h)` is the `(y0, inputs)` the parameter moved by `h` gives.
    fn assert_matches_fd(
        prepared: &PreparedSolver,
        set: InputSet<'_>,
        perturbed: impl Fn(f64) -> (Vec<f64>, Vec<f64>),
    ) -> SolveOutcome {
        let (_, _, t_eval) = small_dae_setup();
        let outcome = prepared
            .solve(SolveRequest::new(&t_eval).with_sensitivities(), set)
            .expect("sensitivity solve failed");
        let h = 1e-6;
        let [plus, minus] = [h, -h].map(|step| {
            let (y0, inputs) = perturbed(step);
            prepared
                .solve(SolveRequest::new(&t_eval), InputSet::new(&y0, &inputs))
                .expect("perturbed solve failed")
        });
        let last = (t_eval.len() - 1) * outcome.n_rows;
        for row in 0..outcome.n_rows {
            let fd = (plus.y[last + row] - minus.y[last + row]) / (2.0 * h);
            let got = blocks(&outcome)[0][last + row];
            assert!((got - fd).abs() < 1e-4, "row {row}: got {got}, fd {fd}");
        }
        outcome
    }

    #[test]
    fn a_sensitivity_request_matches_finite_difference_on_inputs() {
        let prepared = build_small_dae_prepared_with_sens();
        let (y0, inputs, _) = small_dae_setup();
        let outcome = assert_matches_fd(&prepared, InputSet::new(&y0, &inputs), |h| {
            (y0.clone(), vec![inputs[0] + h])
        });
        assert!(!outcome.statistics.sens_error_control_relaxed);
    }

    #[test]
    fn a_subset_sens_solve_integrates_only_the_requested_column() {
        // [b] is a non-prefix subset: it catches a truncating seed that [a] would hide.
        let (y0, inputs) = ([1.0, 0.0], [1.0, 0.5]);
        for param in [0, 1] {
            let model = build_small_dae_two_inputs(&[ParamIdx::from_usize(param)]);
            let prepared = PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("prepare");
            let outcome = assert_matches_fd(&prepared, InputSet::new(&y0, &inputs), |h| {
                let mut moved = inputs.to_vec();
                moved[param] += h;
                (y0.to_vec(), moved)
            });
            assert_eq!(blocks(&outcome).len(), 1, "2 inputs, 1 requested");
        }
    }

    #[test]
    fn seeded_y0_sens_matches_finite_difference_through_the_initial_condition() {
        // y0 itself is `a`, so d/da picks up the seed as well as the rhs term.
        let prepared = build_small_dae_prepared_with_sens();
        let a = 1.0;
        let (y0, inputs) = ([a, a], [a]);
        let seeded = assert_matches_fd(
            &prepared,
            InputSet::new(&y0, &inputs).with_sens_seed(&[1.0, 1.0]),
            |h| (vec![a + h, a + h], vec![a + h]),
        );

        let (_, _, t_eval) = small_dae_setup();
        let unseeded = prepared
            .solve(
                SolveRequest::new(&t_eval).with_sensitivities(),
                InputSet::new(&y0, &inputs),
            )
            .expect("unseeded sensitivity solve failed");
        let last = (t_eval.len() - 1) * seeded.n_rows;
        assert!(
            (blocks(&unseeded)[0][last] - blocks(&seeded)[0][last]).abs() > 1e-3,
            "the seed has to move the answer, else the test proves nothing",
        );
    }

    #[test]
    fn a_sensitivity_request_without_sensitivity_params_is_an_error() {
        // Reported as itself, not via the relaxed retry, and ahead of the seed and row space.
        let prepared = PreparedSolver::new(decay(), 1e-8, &[1e-8]).expect("prepare");
        let request = SolveRequest::new(&[0.0, 0.1]).with_sensitivities();
        let set = InputSet::new(&[1.0], &[]);
        for (request, set) in [
            (request, set),
            (request, set.with_sens_seed(&[0.5])),
            (request.with_outputs(), set),
        ] {
            let err = prepared
                .solve(request, set)
                .expect_err("a sensitivity request the model cannot supply must be an error");
            assert!(
                matches!(err, CoreError::NoSensitivityParams),
                "{request:?}, {set:?}: got {err:?}"
            );
        }
    }

    /// The shared decay fixture, prepared at tolerances tight enough for analytic checks.
    fn build_decay_param_output_event() -> PreparedSolver {
        PreparedSolver::new(decay_with_output_and_event(), 1e-10, &[1e-10])
            .expect("PreparedSolver failed")
    }

    /// Two states, three outputs, one sensitivity parameter and an event: row spaces differ.
    fn build_two_state_three_output_with_sens() -> PreparedSolver {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let two = arena.alloc(Node::Scalar(2.0));
        let neg_two_a = arena.alloc(Node::Mul(neg_a, two));
        let rhs = {
            let d0 = arena.alloc(Node::Mul(neg_a, y0));
            let d1 = arena.alloc(Node::Mul(neg_two_a, y1));
            arena.alloc(Node::Concat(vec![d0, d1]))
        };
        let mass = identity_mass_matrix(2);
        let mut model = compile_with_sens(&arena, rhs, mass, 2, 1, &[ParamIdx::new(0)]);
        let out0 = arena.alloc(Node::Mul(two, y0));
        let out1 = arena.alloc(Node::Add(y0, y1));
        let out2 = arena.alloc(Node::Mul(two, y1));
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[out0, out1, out2])
            .expect("valid outputs");
        let threshold = arena.alloc(Node::Scalar(0.4));
        let event = arena.alloc(Node::Sub(y0, threshold));
        model
            .set_observables(ObservableKind::Events, &mut arena, &[event])
            .expect("a valid event");

        PreparedSolver::new(model, 1e-10, &[1e-10, 1e-10])
            .expect("PreparedSolver failed")
            .with_store_yp(true)
    }

    #[test]
    fn every_payload_combination_honours_one_layout_contract() {
        let prepared = build_two_state_three_output_with_sens();
        let (n_states, n_outputs) = (2, 3);
        // Runs past the event at y0 = 0.4, so every combination terminates on a root.
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
                assert_eq!(
                    outcome.y.len(),
                    outcome.n_rows * outcome.t.len(),
                    "{label}: trajectory size"
                );

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

                let Termination::Event { t: t_event } = outcome.termination else {
                    panic!("{label}: the event should have fired");
                };
                assert!(
                    (t_event - outcome.t[outcome.t.len() - 1]).abs() < 1e-12,
                    "{label}: the trajectory should end at the root"
                );
                let y_event = outcome.y_event.as_ref().expect("y_event missing");
                assert_eq!(y_event.len(), n_states, "{label}: y_event is a full state");
                if outputs {
                    // The root column holds the outputs of the very state y_event reports.
                    let root = &outcome.y[outcome.y.len() - n_outputs..];
                    let (y0, y1) = (y_event[0], y_event[1]);
                    assert_eq!(root, [2.0 * y0, y0 + y1, 2.0 * y1], "{label}");
                }
            }
        }
    }

    #[test]
    // Bit-identical is the property under test.
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
        assert_eq!(states.t.len(), outputs.t.len());
        assert_eq!(states.termination, outputs.termination);
        assert_eq!(states.y_event, outputs.y_event);
        assert_eq!(
            states.statistics.solver.number_of_steps,
            outputs.statistics.solver.number_of_steps
        );
    }

    #[test]
    // Bit-identical is the property under test: doubling is exact.
    #[allow(clippy::float_cmp)]
    fn an_outputs_request_keeps_every_column_across_batch_windows() {
        // Two full windows and a partial one, ending before the event at t = ln 2.5.
        let prepared = build_decay_param_output_event();
        let n_points = 2 * OUTPUT_BATCH_LANES + 3;
        let t_eval: Vec<f64> = (0..n_points).map(|i| i as f64 * 1e-3).collect();
        let set = InputSet::new(&[1.0], &[1.0]);

        let states = prepared
            .solve(SolveRequest::new(&t_eval), set)
            .expect("state solve failed");
        let outputs = prepared
            .solve(SolveRequest::new(&t_eval).with_outputs(), set)
            .expect("output solve failed");

        assert_eq!(outputs.termination, Termination::Completed);
        assert_eq!(outputs.t, t_eval);
        let doubled: Vec<f64> = states.y.iter().map(|y| 2.0 * y).collect();
        assert_eq!(outputs.y, doubled);
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

        assert_eq!(r.termination, Termination::Completed);
        assert_eq!(r.n_rows, 1);
        assert_eq!(blocks(&r).len(), 1);
        assert_eq!(blocks(&r)[0].len(), r.t.len());
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
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[1.0]))
            .expect("solve failed");

        let Termination::Event { t: t_event } = r.termination else {
            panic!("expected event termination");
        };
        assert!((t_event - 2.5f64.ln()).abs() < 1e-6, "t_event = {t_event}");
        let drained = t_eval.iter().filter(|&&t| t <= t_event).count();
        assert_eq!(
            r.t.len(),
            drained + 1,
            "one root column past the grid points"
        );
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
    fn every_solve_path_reports_its_own_integration_time() {
        let prepared = build_decay_param_output_event();
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.1).collect();
        let base = SolveRequest::new(&t_eval);
        for request in [
            base,
            base.with_outputs(),
            base.with_sensitivities(),
            base.with_outputs().with_sensitivities(),
        ] {
            let statistics = prepared
                .solve(request, InputSet::new(&[1.0], &[1.0]))
                .expect("solve")
                .statistics;
            assert!(
                statistics.integration_time_secs > 0.0,
                "{request:?} reported no integration time",
            );
            assert!(
                statistics.integration_time_secs
                    >= statistics.ic_time_secs + statistics.solver_setup_time_secs,
                "{request:?} reported less time than the phases it contains",
            );
        }
    }

    fn grid(n: usize) -> Vec<f64> {
        assert!(n > 1, "a grid needs at least two columns");
        let last = (n - 1) as f64;
        (0..n).map(|i| i as f64 / last).collect()
    }

    #[test]
    fn yp_is_stored_up_to_the_hermite_column_limit_on_every_state_path() {
        let prepared = build_small_dae_prepared_with_sens().with_store_yp(true);
        let (y0, inputs, _) = small_dae_setup();
        let (at, past) = (grid(MAX_HERMITE_COLUMNS), grid(MAX_HERMITE_COLUMNS + 1));
        for (request, stores_yp) in [
            (SolveRequest::new(&at), true),
            (SolveRequest::new(&past), false),
            (SolveRequest::new(&past).with_sensitivities(), false),
        ] {
            let result = prepared
                .solve(request, InputSet::new(&y0, &inputs))
                .expect("solve");
            let columns = request.t_eval.len();
            assert_eq!(result.t.len(), columns, "every column is still reported");
            assert_eq!(result.yp.is_some(), stores_yp, "{columns} columns");
        }
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
    fn the_root_column_carries_the_wound_back_derivative() {
        // yp must be the root-time slope after state_mut_back, not the overshot step's.
        let prepared = build_decay_param_output_event().with_store_yp(true);
        let t_eval: Vec<f64> = (0..=20).map(|i| f64::from(i) * 0.1).collect();
        let r = prepared
            .solve(SolveRequest::new(&t_eval), InputSet::new(&[1.0], &[1.0]))
            .expect("solve failed");
        assert!(matches!(r.termination, Termination::Event { .. }));
        let yp = r.yp.expect("yp requested");
        let root = r.t.len() - 1;
        assert!(
            (yp[root] - (-0.4)).abs() < 1e-6,
            "yp at the y = 0.4 root is {}, want -0.4",
            yp[root],
        );
    }

    #[test]
    fn an_outputs_sensitivity_request_ends_at_the_event_root() {
        // Runs past the event at y = 0.4 (t = ln 2.5), where dy/da = -t y.
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

        let Termination::Event { t: t_event } = r.termination else {
            panic!("expected event termination");
        };
        let ln2p5 = 2.5f64.ln();
        assert!(
            (t_event - ln2p5).abs() < 1e-6,
            "t_event={t_event}, want {ln2p5}"
        );
        assert_eq!(blocks(&r)[0].len(), r.t.len());
        let y_event = r.y_event.as_deref().expect("y_event missing");
        assert!(
            (y_event[0] - 0.4).abs() < 1e-6,
            "y_event={}, expected state 0.4",
            y_event[0]
        );
        let out_last = r.y[r.t.len() - 1];
        assert!((out_last - 0.8).abs() < 1e-6, "out at root={out_last}");
        let sens_last = blocks(&r)[0][r.t.len() - 1];
        let expected = -2.0 * t_event * 0.4;
        assert!(
            (sens_last - expected).abs() < 1e-4,
            "d(out)/da at root={sens_last}, want {expected}"
        );

        let start = r.y_start_sens.as_deref().expect("y_start_sens missing");
        let end = r.y_event_sens.as_deref().expect("y_event_sens missing");
        assert_eq!((start.len(), start[0].len()), (1, 1));
        assert_eq!((end.len(), end[0].len()), (1, 1));
        assert!(start[0][0].abs() < 1e-12, "dy/da at start={}", start[0][0]);
        let expected = -t_event * 0.4;
        assert!(
            (end[0][0] - expected).abs() < 1e-4,
            "dy/da at root={}, want {expected}",
            end[0][0]
        );
    }

    #[test]
    fn end_state_sensitivities_follow_y_event() {
        // Ends before the event: only the outputs request reports a (terminal) y_event.
        let prepared = build_decay_param_output_event();
        let t_eval = [0.0, 0.25, 0.5];
        for outputs in [false, true] {
            let request = SolveRequest::new(&t_eval).with_sensitivities();
            let request = if outputs {
                request.with_outputs()
            } else {
                request
            };
            let r = prepared
                .solve(request, InputSet::new(&[1.0], &[1.0]))
                .expect("sensitivity solve failed");
            assert!(r.y_start_sens.is_some());
            assert_eq!(r.y_event.is_some(), outputs);
            assert_eq!(r.y_event_sens.is_some(), outputs);
            if let Some(end) = r.y_event_sens.as_deref() {
                let expected = -0.5 * (-0.5f64).exp();
                assert!(
                    (end[0][0] - expected).abs() < 1e-6,
                    "dy/da at t=0.5 is {}, want {expected}",
                    end[0][0]
                );
            }
        }

        let r = prepared
            .solve(
                SolveRequest::new(&t_eval).with_outputs(),
                InputSet::new(&[1.0], &[1.0]),
            )
            .expect("outputs solve failed");
        assert!(r.y_start_sens.is_none() && r.y_event_sens.is_none());
    }

    /// `(ws, eqn)` over the two-input fixture with the given sensitivity subset.
    fn two_input_rhs_at_y0(sens: &[ParamIdx]) -> (Rc<RefCell<Workspace>>, Equations) {
        let model = build_small_dae_two_inputs(sens);
        let prepared =
            PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
        assert_eq!(prepared.compiled.n_sens_params(), sens.len());
        let ws = Rc::new(RefCell::new(prepared.compiled.create_workspace()));
        let eqn = prepared.build_eqn(
            InputSet::new(&[1.0, 0.0], &[1.0, 0.5]),
            &ws,
            &prepared.sens_binding,
            Par::Seq,
        );
        (ws, eqn)
    }

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
        // Reordered subset [b, a]: column order follows the subset, not the input vector.
        let (_ws, eqn) = two_input_rhs_at_y0(&[ParamIdx::new(1), ParamIdx::new(0)]);
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
    fn rhs_sens_inplace_matches_per_column_sens_mul_on_a_subset() {
        let (_ws, eqn) = two_input_rhs_at_y0(&[ParamIdx::new(1), ParamIdx::new(0)]);
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
            expected[j * n..(j + 1) * n].copy_from_slice(lane(&col));
            v.set_index(j, 0.0);
        }

        let mut m = FaerSparseMat::<f64>::new_from_sparsity(n, np, rhs.sens_sparsity(), ctx);
        rhs.sens_inplace(&x, 0.0, &mut m);

        assert_eq!(dense_columns(&m, n, np), expected);
    }

    #[test]
    fn sens_inplace_runs_one_primal_pass_regardless_of_k() {
        let one = [ParamIdx::new(0)];
        let two = [ParamIdx::new(1), ParamIdx::new(0)];
        for sens in [&one[..], &two[..]] {
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
    fn sens_atol_tightens_only_differential_states() {
        let model = build_small_dae_two_inputs(&[ParamIdx::new(0)]);
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
    fn with_sens_atol_factor_rejects_non_positive_and_non_finite() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let model = build_small_dae_two_inputs(&[ParamIdx::new(0)]);
            let prepared =
                PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("PreparedSolver failed");
            assert!(
                prepared.with_sens_atol_factor(bad).is_err(),
                "factor {bad} should be rejected"
            );
        }
    }

    #[test]
    fn a_sensitivity_solve_reports_an_initial_condition_failure_as_itself() {
        // `0 = z^2 + 1` has no real root, whatever the sensitivity tolerances are.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let z = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let neg = arena.alloc(Node::Scalar(-1.0));
        let neg_a = arena.alloc(Node::Mul(neg, a));
        let dydt = arena.alloc(Node::Mul(neg_a, y));
        let one = arena.alloc(Node::Scalar(1.0));
        let z2 = arena.alloc(Node::Mul(z, z));
        let constraint = arena.alloc(Node::Add(z2, one));
        let rhs = arena.alloc(Node::Concat(vec![dydt, constraint]));
        let mass = differential_then_algebraic_mass();
        let model = compile_with_sens(&arena, rhs, mass, 2, 1, &[ParamIdx::new(0)]);
        let prepared = PreparedSolver::new(model, 1e-8, &[1e-8, 1e-8]).expect("prepare");
        let err = prepared
            .solve(
                SolveRequest::new(&[0.0, 1.0]).with_sensitivities(),
                InputSet::new(&[1.0, 0.0], &[1.0]),
            )
            .expect_err("an inconsistent IC must fail");
        assert!(
            matches!(err, CoreError::Diffsol(_)),
            "expected the IC failure itself, got: {err:?}"
        );
    }

    /// One sensitivity integration, with or without the sensitivities in error control.
    ///
    /// `relax_after_ic` clears a controlled problem's tolerances after its IC, as the retry does.
    fn sens_attempt(
        prepared: &PreparedSolver,
        t_eval: &[f64],
        set: InputSet<'_>,
        controlled: bool,
        relax_after_ic: bool,
    ) -> SolveOutcome {
        let times = TimePlan::new(t_eval, &[]).expect("sorted grid");
        let (ws, mut problem) = prepared
            .build_problem(
                set,
                times.t0(),
                &prepared.sens_binding,
                controlled,
                Par::Seq,
            )
            .expect("problem");
        let state = problem.bdf_state_sens::<SparseLu>().expect("IC");
        if relax_after_ic {
            problem.sens_rtol = None;
            problem.sens_atol = None;
        }
        let run = Run {
            times,
            ws: &ws,
            inputs: set.inputs,
            with_output: false,
            started: Instant::now(),
        };
        prepared
            .integrate_sensitivities(&problem, state, run, SolverStatistics::default())
            .expect("attempt")
    }

    #[test]
    // Bit-identical is the property under test.
    #[allow(clippy::float_cmp)]
    fn the_retry_from_the_shared_ic_matches_a_relaxed_solve_from_scratch() {
        let model = build_small_dae_two_inputs(&[ParamIdx::new(0)]);
        let prepared = PreparedSolver::new(model, 1e-10, &[1e-10, 1e-10]).expect("prepare");
        let t_eval: Vec<f64> = (0..=5).map(|i| f64::from(i) * 0.2).collect();
        let set = InputSet::new(&[1.0, 0.0], &[1.0, 0.5]);

        let fresh = sens_attempt(&prepared, &t_eval, set, false, false);
        let shared = sens_attempt(&prepared, &t_eval, set, true, true);
        assert_eq!(shared.t, fresh.t);
        assert_eq!(shared.y, fresh.y);
        assert_eq!(shared.sensitivities, fresh.sensitivities);
        assert_eq!(shared.y_start, fresh.y_start);
        assert_eq!(shared.y_start_sens, fresh.y_start_sens);
        assert_eq!(
            shared.statistics.solver.number_of_steps,
            fresh.statistics.solver.number_of_steps
        );

        let controlled = sens_attempt(&prepared, &t_eval, set, true, false);
        assert_ne!(
            controlled.sensitivities, fresh.sensitivities,
            "error control must move the answer, else clearing it proves nothing"
        );
        // Out of error control, the sensitivity is still integrated, not dropped.
        let last = (t_eval.len() - 1) * fresh.n_rows;
        let (controlled_s, relaxed_s) = (blocks(&controlled)[0][last], blocks(&fresh)[0][last]);
        assert!(
            (controlled_s - relaxed_s).abs() < 1e-5 * controlled_s.abs().max(1e-8),
            "relaxed sens {relaxed_s} diverged from controlled {controlled_s}"
        );
    }
}
