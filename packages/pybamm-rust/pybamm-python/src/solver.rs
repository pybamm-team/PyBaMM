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
use pybamm_core::solver::solve as core_solve;

use crate::errors::core_err_to_py;
use crate::model::CompiledModel;
use crate::rayon_pool::pool_for;

/// Declare `PySolverOptions` and both conversions from core's option list, so the
/// defaults dict carries exactly the keys an `options` dict must supply.
macro_rules! solver_options {
    ($($(#[$doc:meta])* $field:ident: $ty:ty,)*) => {
        /// Integrator tuning from `PyBaMM`'s `options` dict; every key is required.
        #[derive(Debug, FromPyObject, IntoPyObject)]
        #[pyo3(from_item_all)]
        pub struct PySolverOptions {
            $($field: $ty,)*
        }

        impl From<PySolverOptions> for SolverOptions {
            fn from(options: PySolverOptions) -> Self {
                Self { $($field: options.$field,)* }
            }
        }

        impl From<SolverOptions> for PySolverOptions {
            fn from(options: SolverOptions) -> Self {
                Self { $($field: options.$field,)* }
            }
        }
    };
}

pybamm_core::for_each_solver_option!(solver_options);

/// The integrator defaults, as the `options` dict `PyBaMM` overlays onto.
#[pyfunction]
pub fn default_solver_options(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    PySolverOptions::from(SolverOptions::default()).into_pyobject(py)
}

/// BDF solver statistics exposed to Python.
#[derive(Debug, Clone)]
// Output-only type: opt out of the automatic `FromPyObject` derive.
#[pyclass(get_all, skip_from_py_object, module = "pybamm.rust")]
pub struct SolverStatistics {
    number_of_steps: usize,
    number_of_linear_solver_setups: usize,
    number_of_nonlinear_solver_iterations: usize,
    number_of_nonlinear_solver_fails: usize,
    number_of_error_test_failures: usize,
    number_of_linear_solver_setups_from_checkpoint: usize,
    number_of_linear_solver_setups_from_first_convergence_fail: usize,
    number_of_linear_solver_setups_from_second_convergence_fail: usize,
    number_of_linear_solver_setups_from_error_test_fail: usize,
    number_of_linear_solver_setups_from_step_success: usize,
    problem_setup_time_secs: f64,
    ic_time_secs: f64,
    solver_setup_time_secs: f64,
    integration_time_secs: f64,
    sens_error_control_relaxed: bool,
}

#[pymethods]
impl SolverStatistics {
    fn __repr__(&self) -> String {
        format!(
            "SolverStatistics(steps={}, linear_setups={} \
             [checkpoint={}, 1st_conv_fail={}, 2nd_conv_fail={}, err_fail={}, heuristic={}], \
             nl_iters={}, nl_fails={}, err_fails={}, \
             problem_setup={:.3}ms, ic_time={:.3}ms, solver_setup={:.3}ms, \
             integration={:.3}ms)",
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
            self.problem_setup_time_secs * 1000.0,
            self.ic_time_secs * 1000.0,
            self.solver_setup_time_secs * 1000.0,
            self.integration_time_secs * 1000.0,
        )
    }
}

impl From<core_solve::SolverStatistics> for SolverStatistics {
    fn from(s: core_solve::SolverStatistics) -> Self {
        let counters = s.solver;
        Self {
            number_of_steps: counters.number_of_steps,
            number_of_linear_solver_setups: counters.number_of_linear_solver_setups,
            number_of_nonlinear_solver_iterations: counters.number_of_nonlinear_solver_iterations,
            number_of_nonlinear_solver_fails: counters.number_of_nonlinear_solver_fails,
            number_of_error_test_failures: counters.number_of_error_test_failures,
            number_of_linear_solver_setups_from_checkpoint: counters
                .number_of_linear_solver_setups_from_checkpoint,
            number_of_linear_solver_setups_from_first_convergence_fail: counters
                .number_of_linear_solver_setups_from_first_convergence_fail,
            number_of_linear_solver_setups_from_second_convergence_fail: counters
                .number_of_linear_solver_setups_from_second_convergence_fail,
            number_of_linear_solver_setups_from_error_test_fail: counters
                .number_of_linear_solver_setups_from_error_test_fail,
            number_of_linear_solver_setups_from_step_success: counters
                .number_of_linear_solver_setups_from_step_success,
            problem_setup_time_secs: s.problem_setup_time_secs,
            ic_time_secs: s.ic_time_secs,
            solver_setup_time_secs: s.solver_setup_time_secs,
            integration_time_secs: s.integration_time_secs,
            sens_error_control_relaxed: s.sens_error_control_relaxed,
        }
    }
}

/// Result of a diffsol solve; payloads the solve was not asked for read as `None`.
#[derive(Debug)]
#[pyclass(module = "pybamm.rust")]
pub struct SolveOutcome {
    /// Solver flag: 0 = success, 1 = root found
    #[pyo3(get)]
    flag: i32,
    /// Time at which an event was triggered, if any
    #[pyo3(get)]
    t_event: Option<f64>,
    #[pyo3(get)]
    statistics: SolverStatistics,
    t_vec: Vec<f64>,
    /// Flat column-major trajectory of states, or of outputs when requested.
    y_flat: Vec<f64>,
    /// Flat row-derivative trajectory matching `y_flat`, when stored.
    yp_flat: Option<Vec<f64>>,
    sens_blocks: Option<Vec<Vec<f64>>>,
    n_rows: usize,
    n_times: usize,
    y_event_vec: Option<Vec<f64>>,
    y_start_vec: Vec<f64>,
    y_start_sens: Option<Vec<Vec<f64>>>,
    y_event_sens: Option<Vec<Vec<f64>>>,
    y_cache: Option<Py<PyArray2<f64>>>,
    yp_cache: Option<Py<PyArray2<f64>>>,
}

#[pymethods]
impl SolveOutcome {
    /// Time points as a numpy array.
    #[getter]
    fn t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.t_vec)
    }

    /// Trajectory matrix `(n_rows, n_times)` of states, or of outputs when requested.
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

    /// Row-derivative matrix `(n_rows, n_times)`, or `None` without `store_yp` or for outputs.
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

    /// Per-parameter sensitivities in the flat layout of `y`, or `None` if not requested.
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

    /// Full state after the initial-condition solve, never an outputs row.
    #[getter]
    fn y_start<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.y_start_vec)
    }

    /// Per-parameter state sensitivities at `y_start`, or `None` if not requested.
    #[getter]
    #[pyo3(name = "yS_start")]
    fn start_sensitivities<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray1<f64>>>> {
        self.y_start_sens
            .as_ref()
            .map(|blocks| sens_blocks(blocks, py))
    }

    /// Per-parameter state sensitivities at `y_event`, or `None` if either is absent.
    #[getter]
    #[pyo3(name = "yS_event")]
    fn event_sensitivities<'py>(&self, py: Python<'py>) -> Option<Vec<Bound<'py, PyArray1<f64>>>> {
        self.y_event_sens
            .as_ref()
            .map(|blocks| sens_blocks(blocks, py))
    }
}

impl From<core_solve::SolveOutcome> for SolveOutcome {
    fn from(outcome: core_solve::SolveOutcome) -> Self {
        Self {
            flag: outcome.termination.as_flag(),
            t_event: outcome.termination.event_time(),
            statistics: SolverStatistics::from(outcome.statistics),
            n_times: outcome.t.len(),
            t_vec: outcome.t,
            y_flat: outcome.y,
            yp_flat: outcome.yp,
            sens_blocks: outcome.sensitivities,
            n_rows: outcome.n_rows,
            y_event_vec: outcome.y_event,
            y_start_vec: outcome.y_start,
            y_start_sens: outcome.y_start_sens,
            y_event_sens: outcome.y_event_sens,
            y_cache: None,
            yp_cache: None,
        }
    }
}

/// Move a column-major flat buffer into a cached `(nrows, ncols)` F-order array.
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
/// A seed without `sensitivities` is rejected rather than silently ignored.
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

/// Absolute tolerance: one value shared by every state, or one per state.
///
/// `PerState` comes first because a length-1 array would also extract as `Uniform`.
#[derive(Debug, FromPyObject)]
pub enum PyAtol<'py> {
    PerState(PyReadonlyArray1<'py, f64>),
    Uniform(f64),
}

impl PyAtol<'_> {
    /// Widen to the per-state vector the core takes.
    fn into_vec(self, n_states: usize) -> Vec<f64> {
        match self {
            Self::PerState(atol) => atol.as_array().to_vec(),
            Self::Uniform(atol) => vec![atol; n_states],
        }
    }
}

/// Prepare-once/execute-many solver for repeated integrations of one model.
#[derive(Debug)]
#[pyclass(frozen, module = "pybamm.rust")]
pub struct PreparedSolver {
    inner: core_solve::PreparedSolver,
}

#[pymethods]
impl PreparedSolver {
    /// Prepare a solver for repeated use with the given model and tolerances.
    ///
    /// `store_yp` also stores the state time derivatives (`result.yp`) for Hermite interpolation.
    #[new]
    // Keep `sens_atol_factor` equal to `core_solve::DEFAULT_SENS_ATOL_FACTOR`; a
    // path default renders as `...` in `__text_signature__`, which stubtest rejects.
    #[pyo3(signature = (model, rtol=1e-6, atol=PyAtol::Uniform(1e-6), sens_atol_factor=1.0, options=None, store_yp=false))]
    fn new(
        py: Python<'_>,
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

        let mut prepared = core_solve::PreparedSolver::new(compiled, rtol, &atol_vec)
            .and_then(|p| p.with_sens_atol_factor(sens_atol_factor))
            .and_then(|p| p.with_options(options))
            .map_err(core_err_to_py)?
            .with_store_yp(store_yp);
        // Only stored slopes read dF/dt, so other solves never compile it.
        if store_yp {
            let graph = model
                .graph
                .try_borrow(py)
                .map_err(|_| PyValueError::new_err("PreparedSolver: graph borrow conflict"))?;
            prepared = prepared.with_time_derivative(graph.arena(), model.rhs_root);
        }

        Ok(Self { inner: prepared })
    }

    /// Solve the model over the given time span.
    ///
    /// Every `t_stop` restart time must also appear in `t_eval`; `y0_sens` is the
    /// flattened, column-major `dy0/dp` seed. `num_threads` workers factorise the LU.
    #[pyo3(signature = (t_eval, t_stop, y0, inputs, *, outputs=false, sensitivities=false, y0_sens=None, num_threads=1))]
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
        num_threads: usize,
    ) -> PyResult<SolveOutcome> {
        let request =
            solve_request(&t_eval, &t_stop, outputs, sensitivities)?.with_lu_threads(num_threads);
        let set = core_solve::InputSet::new(y0.as_slice()?, inputs.as_slice()?)
            .with_sens_seed(y0_sens_slice(y0_sens.as_ref(), sensitivities)?);

        let inner = &self.inner;
        let outcome = py
            .detach(|| {
                if num_threads <= 1 {
                    return Ok(inner.solve(request, set));
                }
                // `solve_batch`'s cached pool, so the LU never exceeds `num_threads`.
                let pool = pool_for(num_threads)?;
                Ok::<_, PyErr>(pool.install(|| inner.solve(request, set)))
            })?
            .map_err(core_err_to_py)?;

        Ok(SolveOutcome::from(outcome))
    }

    /// Solve every input set in `y0`/`inputs`, `num_threads` at a time.
    ///
    /// Arrays are C-contiguous with one row per set; returns one result per row, or the
    /// unraised exception for a failed set.
    #[pyo3(signature = (t_eval, t_stop, y0, inputs, num_threads, *, outputs=false, sensitivities=false, y0_sens=None))]
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
        lu_threads: 1,
    })
}

/// One borrowed row per input set of a C-contiguous `(n_sets, width)` array.
fn batch_rows<'a>(array: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Vec<&'a [f64]>> {
    let shape = array.shape();
    let (n_sets, width) = (shape[0], shape[1]);
    // `as_slice` enforces C-contiguity, which makes the row split sound.
    let flat = array.as_slice()?;
    Ok((0..n_sets)
        .map(|i| &flat[i * width..(i + 1) * width])
        .collect())
}

/// The per-set `dy0/dp` seeds, or `n_sets` empty (all-zero) seeds when omitted.
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
