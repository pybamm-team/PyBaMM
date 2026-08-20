// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pybamm_core::CoreError;
use pybamm_core::solver::SolverOptions;
use pybamm_core::solver::batch::check_batch_widths;
// Core and binding share one name per concept, so the boundary crossing is
// spelled out at every use site rather than hidden in a mutated noun.
use pybamm_core::solver::solve as core_solve;

use crate::errors::core_err_to_py;
use crate::model::CompiledModel;
use crate::pool::pool_for;

/// Integrator tuning as `PyBaMM`'s `options` dict, extracted item by item.
///
/// Every field is required: the Python solver owns the defaults and always
/// sends a complete dict, so a key missing here is a wiring bug rather than
/// something to paper over with a fallback.
#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct PySolverOptions {
    max_nonlinear_solver_iterations: usize,
    max_error_test_failures: usize,
    max_nonlinear_solver_failures: usize,
    nonlinear_solver_tolerance: f64,
    min_timestep: f64,
    max_timestep_growth: Option<f64>,
    min_timestep_growth: Option<f64>,
    max_timestep_shrink: Option<f64>,
    min_timestep_shrink: Option<f64>,
    update_jacobian_after_steps: usize,
    update_rhs_jacobian_after_steps: usize,
    threshold_to_update_jacobian: f64,
    threshold_to_update_rhs_jacobian: f64,
    pi_control_proportional: f64,
    pi_control_integral: f64,
}

/// The integrator defaults, as the `options` dict `PyBaMM` overlays onto.
///
/// Exposed so `pybamm.DiffsolSolver.DEFAULT_OPTIONS` can be pinned against
/// diffsol's own defaults rather than trusted to stay a faithful hand-copy.
#[pyfunction]
pub fn default_solver_options(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let defaults = SolverOptions::default();
    let dict = PyDict::new(py);
    dict.set_item(
        "max_nonlinear_solver_iterations",
        defaults.max_nonlinear_solver_iterations,
    )?;
    dict.set_item("max_error_test_failures", defaults.max_error_test_failures)?;
    dict.set_item(
        "max_nonlinear_solver_failures",
        defaults.max_nonlinear_solver_failures,
    )?;
    dict.set_item(
        "nonlinear_solver_tolerance",
        defaults.nonlinear_solver_tolerance,
    )?;
    dict.set_item("min_timestep", defaults.min_timestep)?;
    dict.set_item("max_timestep_growth", defaults.max_timestep_growth)?;
    dict.set_item("min_timestep_growth", defaults.min_timestep_growth)?;
    dict.set_item("max_timestep_shrink", defaults.max_timestep_shrink)?;
    dict.set_item("min_timestep_shrink", defaults.min_timestep_shrink)?;
    dict.set_item(
        "update_jacobian_after_steps",
        defaults.update_jacobian_after_steps,
    )?;
    dict.set_item(
        "update_rhs_jacobian_after_steps",
        defaults.update_rhs_jacobian_after_steps,
    )?;
    dict.set_item(
        "threshold_to_update_jacobian",
        defaults.threshold_to_update_jacobian,
    )?;
    dict.set_item(
        "threshold_to_update_rhs_jacobian",
        defaults.threshold_to_update_rhs_jacobian,
    )?;
    dict.set_item("pi_control_proportional", defaults.pi_control_proportional)?;
    dict.set_item("pi_control_integral", defaults.pi_control_integral)?;
    Ok(dict.unbind())
}

impl From<PySolverOptions> for SolverOptions {
    fn from(options: PySolverOptions) -> Self {
        Self {
            max_nonlinear_solver_iterations: options.max_nonlinear_solver_iterations,
            max_error_test_failures: options.max_error_test_failures,
            max_nonlinear_solver_failures: options.max_nonlinear_solver_failures,
            nonlinear_solver_tolerance: options.nonlinear_solver_tolerance,
            min_timestep: options.min_timestep,
            max_timestep_growth: options.max_timestep_growth,
            min_timestep_growth: options.min_timestep_growth,
            max_timestep_shrink: options.max_timestep_shrink,
            min_timestep_shrink: options.min_timestep_shrink,
            update_jacobian_after_steps: options.update_jacobian_after_steps,
            update_rhs_jacobian_after_steps: options.update_rhs_jacobian_after_steps,
            threshold_to_update_jacobian: options.threshold_to_update_jacobian,
            threshold_to_update_rhs_jacobian: options.threshold_to_update_rhs_jacobian,
            pi_control_proportional: options.pi_control_proportional,
            pi_control_integral: options.pi_control_integral,
        }
    }
}

/// BDF solver statistics exposed to Python.
#[derive(Debug, Clone)]
// Output-only type: opt out of the (now deprecated) automatic FromPyObject derive.
#[pyclass(skip_from_py_object, module = "pybamm.rust")]
pub struct SolverStatistics {
    #[pyo3(get)]
    number_of_steps: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups: usize,
    #[pyo3(get)]
    number_of_nonlinear_solver_iterations: usize,
    #[pyo3(get)]
    number_of_nonlinear_solver_fails: usize,
    #[pyo3(get)]
    number_of_error_test_failures: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups_from_checkpoint: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups_from_first_convergence_fail: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups_from_second_convergence_fail: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups_from_error_test_fail: usize,
    #[pyo3(get)]
    number_of_linear_solver_setups_from_step_success: usize,
    #[pyo3(get)]
    ic_time_secs: f64,
    #[pyo3(get)]
    solver_setup_time_secs: f64,
    #[pyo3(get)]
    integration_time_secs: f64,
    #[pyo3(get)]
    sens_error_control_relaxed: bool,
}

#[pymethods]
impl SolverStatistics {
    fn __repr__(&self) -> String {
        format!(
            "SolverStatistics(steps={}, linear_setups={} \
             [checkpoint={}, 1st_conv_fail={}, 2nd_conv_fail={}, err_fail={}, heuristic={}], \
             nl_iters={}, nl_fails={}, err_fails={}, \
             ic_time={:.3}ms, solver_setup={:.3}ms, integration={:.3}ms)",
            self.number_of_steps,
            self.number_of_linear_solver_setups,
            self.number_of_linear_solver_setups_from_checkpoint,
            self.number_of_linear_solver_setups_from_first_convergence_fail,
            self.number_of_linear_solver_setups_from_second_convergence_fail,
            self.number_of_linear_solver_setups_from_error_test_fail,
            self.number_of_linear_solver_setups_from_step_success,
            self.number_of_nonlinear_solver_iterations,
            self.number_of_nonlinear_solver_fails,
            self.number_of_error_test_failures,
            self.ic_time_secs * 1000.0,
            self.solver_setup_time_secs * 1000.0,
            self.integration_time_secs * 1000.0,
        )
    }
}

impl From<core_solve::SolverStatistics> for SolverStatistics {
    fn from(s: core_solve::SolverStatistics) -> Self {
        Self {
            number_of_steps: s.number_of_steps,
            number_of_linear_solver_setups: s.number_of_linear_solver_setups,
            number_of_nonlinear_solver_iterations: s.number_of_nonlinear_solver_iterations,
            number_of_nonlinear_solver_fails: s.number_of_nonlinear_solver_fails,
            number_of_error_test_failures: s.number_of_error_test_failures,
            number_of_linear_solver_setups_from_checkpoint: s
                .number_of_linear_solver_setups_from_checkpoint,
            number_of_linear_solver_setups_from_first_convergence_fail: s
                .number_of_linear_solver_setups_from_first_convergence_fail,
            number_of_linear_solver_setups_from_second_convergence_fail: s
                .number_of_linear_solver_setups_from_second_convergence_fail,
            number_of_linear_solver_setups_from_error_test_fail: s
                .number_of_linear_solver_setups_from_error_test_fail,
            number_of_linear_solver_setups_from_step_success: s
                .number_of_linear_solver_setups_from_step_success,
            ic_time_secs: s.ic_time_secs,
            solver_setup_time_secs: s.solver_setup_time_secs,
            integration_time_secs: s.integration_time_secs,
            sens_error_control_relaxed: s.sens_error_control_relaxed,
        }
    }
}

/// Result of a diffsol solve, exposed to Python.
///
/// Wraps core's `SolveOutcome` and converts data to numpy arrays on access. One
/// type for every payload combination: `y` holds whichever rows the solve was
/// asked for, and the payloads it was not asked for read as `None`.
#[derive(Debug)]
#[pyclass(module = "pybamm.rust")]
pub struct SolveOutcome {
    /// Solver flag: 0 = success, 1 = root found
    #[pyo3(get)]
    flag: i32,
    /// Time at which an event was triggered, if any
    #[pyo3(get)]
    t_event: Option<f64>,
    /// BDF solver statistics
    #[pyo3(get)]
    statistics: SolverStatistics,
    /// Time points returned by the solver
    t_vec: Vec<f64>,
    /// Flat trajectory in column-major order: states, or output variables when
    /// the solve was asked for them.
    y_flat: Vec<f64>,
    /// Flat row-derivative trajectory matching `y_flat`, when stored.
    yp_flat: Option<Vec<f64>>,
    /// Per-parameter sensitivity blocks, `None` when none were requested.
    sens_blocks: Option<Vec<Vec<f64>>>,
    n_rows: usize,
    n_times: usize,
    /// Full state where the trajectory ends, if the solve reported one
    y_event_vec: Option<Vec<f64>>,
    /// Cached y matrix to avoid repeated allocation/copy
    y_cache: Option<Py<PyArray2<f64>>>,
    /// Cached yp matrix to avoid repeated allocation/copy
    yp_cache: Option<Py<PyArray2<f64>>>,
}

#[pymethods]
impl SolveOutcome {
    /// Time points as a numpy array.
    #[getter]
    fn t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.t_vec)
    }

    /// Trajectory matrix with shape `(n_rows, n_times)`.
    ///
    /// Rows are states, or the model's output variables when the solve was asked
    /// for `outputs`; row index varies along rows, time along columns.
    #[getter]
    fn y<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        take_cached_matrix(
            py,
            &mut self.y_cache,
            &mut self.y_flat,
            self.n_rows,
            self.n_times,
        )
    }

    /// Row-derivative matrix with shape `(n_rows, n_times)`, or `None` when the
    /// solver was built with `store_yp=False` or the solve returned outputs.
    #[getter]
    fn yp<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        take_cached_matrix_opt(
            py,
            &mut self.yp_cache,
            &mut self.yp_flat,
            self.n_rows,
            self.n_times,
        )
    }

    /// Per-parameter sensitivities, each matching the flat layout of `y`, or
    /// `None` when the solve was not asked for them.
    #[getter]
    #[pyo3(name = "yS")]
    fn param_sensitivities<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray1<f64>>>> {
        self.sens_blocks
            .as_ref()
            .map(|blocks| sens_blocks(blocks, py))
    }

    /// Full state where the trajectory ends, never an outputs row.
    #[getter]
    fn y_event<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.y_event_vec
            .as_ref()
            .map(|v| PyArray1::from_slice(py, v))
    }
}

impl From<core_solve::SolveOutcome> for SolveOutcome {
    fn from(outcome: core_solve::SolveOutcome) -> Self {
        Self {
            flag: outcome.flag,
            t_event: outcome.t_event,
            statistics: SolverStatistics::from(outcome.statistics),
            t_vec: outcome.t,
            y_flat: outcome.y,
            yp_flat: outcome.yp,
            sens_blocks: outcome.sensitivities,
            n_rows: outcome.n_rows,
            n_times: outcome.n_times,
            y_event_vec: outcome.y_event,
            y_cache: None,
            yp_cache: None,
        }
    }
}

/// Consume a column-major flat buffer into a cached `(nrows, ncols)` F-order
/// array: zero-copy on first access, cached reference thereafter.
fn take_cached_matrix<'py>(
    py: Python<'py>,
    cache: &mut Option<Py<PyArray2<f64>>>,
    flat: &mut Vec<f64>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if let Some(cached) = cache {
        return Ok(cached.bind(py).clone());
    }
    let data = std::mem::take(flat);
    // Not a double-access guard: the cache above makes that unreachable. This
    // fires only when the result was built with empty data for a non-empty
    // shape, which is a producer bug rather than anything the caller did.
    if data.is_empty() && nrows > 0 && ncols > 0 {
        return Err(PyRuntimeError::new_err(format!(
            "result constructed with empty data for a {nrows}x{ncols} matrix"
        )));
    }
    let arr = Array2::from_shape_vec((nrows, ncols).f(), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let py_arr = arr.into_pyarray(py);
    *cache = Some(py_arr.clone().unbind());
    Ok(py_arr)
}

/// [`take_cached_matrix`] for an optional flat buffer: `None` stays `None`.
fn take_cached_matrix_opt<'py>(
    py: Python<'py>,
    cache: &mut Option<Py<PyArray2<f64>>>,
    flat: &mut Option<Vec<f64>>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    flat.as_mut().map_or_else(
        || Ok(None),
        |flat| take_cached_matrix(py, cache, flat, nrows, ncols).map(Some),
    )
}

/// The `dy0/dp` seed as a flat slice; a missing seed is the empty (all-zero) one.
///
/// A seed handed to a solve that was not asked for sensitivities is rejected
/// rather than ignored: the two arguments are one intent, and silently dropping
/// the seed would return zero sensitivities that look computed.
fn y0_sens_slice<'a>(
    y0_sens: Option<&'a PyReadonlyArray1<'_, f64>>,
    sensitivities: bool,
) -> PyResult<&'a [f64]> {
    match y0_sens {
        Some(_) if !sensitivities => Err(seed_without_sensitivities()),
        Some(arr) => Ok(arr.as_slice()?),
        None => Ok(&[]),
    }
}

/// The rejection both seed helpers raise.
fn seed_without_sensitivities() -> PyErr {
    PyValueError::new_err("y0_sens was given but sensitivities=False")
}

/// Per-parameter flat sensitivity blocks as a list of 1-D numpy arrays.
fn sens_blocks<'py>(blocks: &[Vec<f64>], py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
    blocks
        .iter()
        .map(|blk| PyArray1::from_slice(py, blk))
        .collect()
}

/// Absolute tolerance as `PyBaMM` sends it: one value shared by every state, or
/// one value per state.
///
/// The per-state arm is tried first because a length-1 array also satisfies the
/// uniform arm, and would then silently broadcast its single entry.
#[derive(Debug, FromPyObject)]
pub enum PyAtol<'py> {
    PerState(PyReadonlyArray1<'py, f64>),
    Uniform(f64),
}

impl PyAtol<'_> {
    /// Widen to the per-state vector the core takes.
    ///
    /// A per-state array passes through; a length mismatch is
    /// `PreparedSolver::new`'s to reject against the model's own state count.
    fn into_vec(self, n_states: usize) -> Vec<f64> {
        match self {
            Self::PerState(atol) => atol.as_array().to_vec(),
            Self::Uniform(atol) => vec![atol; n_states],
        }
    }
}

/// Prepare-once/execute-many solver for repeated integrations of one model.
///
/// Built once during setup, then driven by many `solve*()` calls with different
/// initial conditions and inputs. `y0`, `t_eval` and the inputs are per call;
/// only the model, tolerances and integrator options are retained.
#[derive(Debug)]
#[pyclass(module = "pybamm.rust")]
pub struct PreparedSolver {
    inner: core_solve::PreparedSolver,
}

#[pymethods]
impl PreparedSolver {
    /// Prepare a solver for repeated use with the given model and tolerances.
    ///
    /// `atol` is either one tolerance for every state or one per state;
    /// `options` is `PyBaMM`'s integrator tuning dict, and omitting it keeps the
    /// defaults in `SolverOptions`. `store_yp` additionally stores the state
    /// time derivatives on the state-trajectory paths (`result.yp`), the knot
    /// slopes cubic-Hermite output interpolation needs.
    #[new]
    #[pyo3(signature = (model, rtol=1e-6, atol=PyAtol::Uniform(1e-6), sens_atol_factor=1e-3, options=None, store_yp=false))]
    fn new(
        model: &CompiledModel,
        rtol: f64,
        atol: PyAtol<'_>,
        sens_atol_factor: f64,
        options: Option<PySolverOptions>,
        store_yp: bool,
    ) -> PyResult<Self> {
        let compiled = Arc::clone(&model.compiled);
        let atol_vec = atol.into_vec(compiled.n_states());
        let options = options.map_or_else(SolverOptions::default, SolverOptions::from);

        let prepared = core_solve::PreparedSolver::new(compiled, rtol, &atol_vec)
            .and_then(|p| p.with_sens_atol_factor(sens_atol_factor))
            .and_then(|p| p.with_options(options))
            .map_err(core_err_to_py)?
            .with_store_yp(store_yp);

        Ok(Self { inner: prepared })
    }

    /// Solve the model over the given time span.
    ///
    /// `t_stop` holds the discontinuity times the integrator must land on
    /// exactly, restarting there; every one of them must also appear in
    /// `t_eval`, which is where the solution is reported.
    ///
    /// `outputs` reports the model's registered output variables instead of the
    /// full state, which cuts the FFI transfer when only a few variables are
    /// wanted, and `sensitivities` adds the forward-sensitivity blocks, seeded by
    /// `y0_sens` (flattened `dy0/dp`, column-major over the requested
    /// parameters). The two compose, so the four payload combinations are these
    /// two flags rather than four entry points.
    #[pyo3(signature = (t_eval, t_stop, y0, inputs, *, outputs=false, sensitivities=false, y0_sens=None))]
    // Over clippy's threshold because pyo3 signatures are flat: the payload flags
    // that replaced four entry points have to be spelled out as arguments.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        t_eval: PyReadonlyArray1<'_, f64>,
        t_stop: PyReadonlyArray1<'_, f64>,
        y0: PyReadonlyArray1<'_, f64>,
        inputs: PyReadonlyArray1<'_, f64>,
        outputs: bool,
        sensitivities: bool,
        y0_sens: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<SolveOutcome> {
        let request = solve_request(&t_eval, &t_stop, outputs, sensitivities)?;
        let set = core_solve::InputSet::new(y0.as_slice()?, inputs.as_slice()?)
            .with_sens_seed(y0_sens_slice(y0_sens.as_ref(), sensitivities)?);

        let inner = &self.inner;
        let outcome = py
            .detach(|| inner.solve(request, set))
            .map_err(core_err_to_py)?;

        Ok(SolveOutcome::from(outcome))
    }

    /// Solve every input set in `y0`/`inputs`, `num_threads` at a time.
    ///
    /// `y0` and `inputs` are C-contiguous 2-D arrays with one row per input set,
    /// and `y0_sens` one seed row per set; `t_eval`, `t_stop` and the payload
    /// flags are shared, as they already are on the callers. The returned list
    /// has one entry per row, in row order: a result object, or — for a set that
    /// failed — the exception *instance*, unraised, which is what keeps the
    /// failing set's identity that one collapsed error would lose.
    ///
    /// One `py.detach()` covers the batch, so Ctrl-C lands when it returns.
    #[pyo3(signature = (t_eval, t_stop, y0, inputs, num_threads, *, outputs=false, sensitivities=false, y0_sens=None))]
    // Over the threshold for the same reason as `solve`, plus the pool width.
    #[allow(clippy::too_many_arguments)]
    fn solve_batch(
        &self,
        py: Python<'_>,
        t_eval: PyReadonlyArray1<'_, f64>,
        t_stop: PyReadonlyArray1<'_, f64>,
        y0: PyReadonlyArray2<'_, f64>,
        inputs: PyReadonlyArray2<'_, f64>,
        num_threads: usize,
        outputs: bool,
        sensitivities: bool,
        y0_sens: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let request = solve_request(&t_eval, &t_stop, outputs, sensitivities)?;
        let y0_rows = batch_rows(&y0)?;
        let inputs_rows = batch_rows(&inputs)?;
        let seed_rows = batch_sens_rows(y0_sens.as_ref(), sensitivities, y0_rows.len())?;
        // The widths a `&[InputSet]` cannot express: three arrays arrive here
        // with independent row counts, and core's rule names the mismatch.
        let seeds_given = y0_sens.is_some().then_some(seed_rows.len());
        check_batch_widths(y0_rows.len(), inputs_rows.len(), seeds_given)
            .map_err(core_err_to_py)?;
        let sets: Vec<core_solve::InputSet<'_>> = y0_rows
            .iter()
            .zip(&inputs_rows)
            .zip(&seed_rows)
            .map(|((y0, inputs), seed)| core_solve::InputSet::new(y0, inputs).with_sens_seed(seed))
            .collect();

        let inner = &self.inner;
        let outcomes = py.detach(|| {
            let pool = pool_for(num_threads)?;
            Ok::<_, PyErr>(pool.install(|| inner.solve_batch(request, &sets)))
        })?;
        batch_entries(py, outcomes)
    }
}

/// Build the shared half of a solve from the arguments Python sent.
fn solve_request<'a>(
    t_eval: &'a PyReadonlyArray1<'_, f64>,
    t_stop: &'a PyReadonlyArray1<'_, f64>,
    outputs: bool,
    sensitivities: bool,
) -> PyResult<core_solve::SolveRequest<'a>> {
    Ok(core_solve::SolveRequest {
        t_eval: t_eval.as_slice()?,
        t_stop: t_stop.as_slice()?,
        outputs,
        sensitivities,
    })
}

/// One borrowed row per input set of a C-contiguous `(n_sets, width)` array.
///
/// A zero-width array (a model with no input parameters) yields `n_sets` empty
/// rows rather than no rows, which is what keeps the batch width equal to the
/// number of sets.
fn batch_rows<'a>(array: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Vec<&'a [f64]>> {
    let shape = array.shape();
    let (n_sets, width) = (shape[0], shape[1]);
    // as_slice enforces C-contiguity, which is what makes the row split sound.
    let flat = array.as_slice()?;
    Ok((0..n_sets)
        .map(|i| &flat[i * width..(i + 1) * width])
        .collect())
}

/// The per-set `dy0/dp` seeds, or `n_sets` empty (all-zero) seeds when omitted.
///
/// Rejects a seed array without `sensitivities` for the same reason
/// [`y0_sens_slice`] does.
fn batch_sens_rows<'a>(
    y0_sens: Option<&'a PyReadonlyArray2<'_, f64>>,
    sensitivities: bool,
    n_sets: usize,
) -> PyResult<Vec<&'a [f64]>> {
    match y0_sens {
        Some(_) if !sensitivities => Err(seed_without_sensitivities()),
        Some(array) => batch_rows(array),
        None => Ok(vec![&[][..]; n_sets]),
    }
}

/// Convert per-set core outcomes into the Python list `solve_batch` returns.
///
/// A failed set contributes its exception instance, built but never raised, so
/// the caller keeps the index alongside the cause.
fn batch_entries(
    py: Python<'_>,
    outcomes: Vec<Result<core_solve::SolveOutcome, CoreError>>,
) -> PyResult<Vec<Py<PyAny>>> {
    outcomes
        .into_iter()
        .map(|outcome| match outcome {
            Ok(value) => Ok(Py::new(py, SolveOutcome::from(value))?.into_any()),
            Err(error) => Ok(core_err_to_py(error).into_value(py).into_any()),
        })
        .collect()
}
