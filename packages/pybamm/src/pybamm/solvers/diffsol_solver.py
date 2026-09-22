"""Solver class wrapping the Rust diffsol BDF integrator."""

import numbers
import sys

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.solvers.base_solver import (
    flatten_inputs,
    validate_rust_sensitivity_widths,
)
from pybamm.solvers.idaklu_solver import _state_sensitivities
from pybamm.solvers.observation import ObserverCache, OutputAssembly
from pybamm.solvers.rust_lowering import RustModelLowering

# Native handles and set-up products that the next set_up() rebuilds
_REBUILDABLE_STATE_KEYS = (
    "_prepared_solver",
    "_rust_model",
    "_output_assembly",
    "_model_set_up",
    "_time_integral_vars",
)


def _as_flat_float64(y0) -> npt.NDArray[np.float64]:
    """Flatten an initial state to a float64 vector (handles ``casadi.DM``)."""
    if hasattr(y0, "full"):
        y0 = y0.full()
    return np.asarray(y0, dtype=np.float64).flatten()


def _flatten_y0_sens(y0S, n_states: int, n_params: int) -> npt.NDArray[np.float64]:
    """Flatten ``dy0/dp`` to the column-major block the Rust solver expects.

    Takes one ``(n_states, 1)`` or ``(n_states,)`` column per parameter, or one
    ``(n_states, n_params)`` matrix; an empty input is the all-zero seed.
    """
    blocks = [y0S] if hasattr(y0S, "full") or isinstance(y0S, np.ndarray) else list(y0S)
    if not blocks:
        # A shape complaint here would mask the solver's clearer error
        return np.zeros(0, dtype=np.float64)
    matrix = np.column_stack(
        [
            np.asarray(
                block.full() if hasattr(block, "full") else block, dtype=np.float64
            )
            for block in blocks
        ]
    )
    if matrix.shape != (n_states, n_params):
        raise pybamm.SolverError(
            f"Initial-condition sensitivities have shape {matrix.shape} but "
            f"({n_states}, {n_params}) was expected (states x sensitivity "
            "parameters)."
        )
    return matrix.ravel(order="F")


class DiffsolSolver(pybamm.BaseSolver):
    """Solve a discretised model using the Rust diffsol BDF solver.

    States are stored only at the output times, not at internal steps. Without
    ``t_interp``, these are ``t_eval`` merged with a uniform 100-point grid over its
    span; an explicit ``t_interp`` (merged with ``t_eval``) sets them exactly.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float or :class:`numpy.ndarray`, optional
        The absolute tolerance, one value or one per state (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The initial-condition method; if None (default), ``"nonlinear_solver"``
        for DAEs when ``calc_ic=False``, otherwise skipped.
    root_tol : float, optional
        The tolerance for the initial-condition solver. Default is 1e-6.
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not.
    on_extrapolation : str, optional
        What to do on extrapolation: "warn" (default), "error" or "ignore".
    on_failure : str, optional
        Only "error" (default): diffsol returns no partial solution, so an
        integration failure always raises :class:`pybamm.SolverError`.
    output_variables : list[str], optional
        Variables to return; if empty (default), the full state is returned.
    calc_ic : bool, optional
        If True (default), diffsol finds consistent initial conditions itself;
        if False, PyBaMM first solves for them with ``root_method``.
    hermite_interpolation : bool, optional
        If True (default), store state derivatives for cubic Hermite off-grid reads.
        Ignored with ``output_variables`` or more than 4096 output points.
    sens_atol_factor : float, optional
        Multiplier on ``atol`` for the differential states' sensitivity tolerance
        (default 1.0, matching :class:`IDAKLUSolver`); lower it for tighter gradients.
    options : dict, optional
        diffsol ``OdeSolverOptions`` overrides plus ``num_threads``: input sets solved
        concurrently, or the threads a single set's sparse LU uses (worthwhile from
        ~50k states). Defaults are in :attr:`DEFAULT_OPTIONS`; unknown keys and
        values of the wrong type raise :class:`pybamm.SolverError`.
    """

    _integrates_via_compiled_model = True
    # By design, a variable that cannot be lowered to Rust hard-fails on read
    _observes_via_compiled_model = True

    #: diffsol's defaults, bar the per-solve ``max_nonlinear_solver_failures``: 50
    #: caps solve length, not divergence. Literal so ``import pybamm`` needs no extension.
    _INTEGRATOR_DEFAULTS = {
        "max_nonlinear_solver_iterations": 10,
        "max_error_test_failures": 40,
        "max_nonlinear_solver_failures": 100000,
        "nonlinear_solver_tolerance": 0.2,
        "min_timestep": 1e-13,
        "max_timestep_growth": None,
        "min_timestep_growth": None,
        "max_timestep_shrink": None,
        "min_timestep_shrink": None,
        "update_jacobian_after_steps": 20,
        "update_rhs_jacobian_after_steps": 50,
        "threshold_to_update_jacobian": 0.3,
        "threshold_to_update_rhs_jacobian": 0.2,
        "pi_control_proportional": 0.0,
        "pi_control_integral": 0.5,
    }

    #: The integrator knobs plus ``num_threads``, which never reaches diffsol.
    DEFAULT_OPTIONS = _INTEGRATOR_DEFAULTS | {"num_threads": 1}

    def __init__(
        self,
        rtol=1e-6,
        atol=1e-6,
        root_method=None,
        root_tol=1e-6,
        extrap_tol=None,
        on_extrapolation=None,
        on_failure=None,
        output_variables=None,
        calc_ic=True,
        hermite_interpolation=True,
        sens_atol_factor=1.0,
        options=None,
    ):
        super().__init__(
            "problem dependent",
            rtol,
            atol,
            root_method,
            root_tol,
            extrap_tol,
            on_extrapolation,
            on_failure,
            output_variables,
        )
        self.name = "diffsol solver (bdf)"
        self._calc_ic = calc_ic
        self._hermite_interpolation = bool(hermite_interpolation)
        self._supports_interp = True
        self._supports_t_eval_discontinuities = True
        self._options = self._combine_options(options)

        try:
            factor = float(sens_atol_factor)
        except (TypeError, ValueError) as exc:
            raise pybamm.SolverError(
                f"sens_atol_factor must be a finite number > 0, got {sens_atol_factor!r}"
            ) from exc
        if not np.isfinite(factor) or factor <= 0:
            raise pybamm.SolverError(
                f"sens_atol_factor must be a finite number > 0, got {sens_atol_factor!r}"
            )
        self._sens_atol_factor = factor

        if root_method is None and not calc_ic:
            self._use_default_root_method = True
        else:
            self._use_default_root_method = False

    @property
    def on_failure(self) -> str:
        return self._on_failure

    @on_failure.setter
    def on_failure(self, value: str) -> None:
        if value in ("warn", "ignore"):
            raise pybamm.SolverError(
                "DiffsolSolver returns no partial solution, so on_failure must be "
                f"'error', not {value!r}"
            )
        pybamm.BaseSolver.on_failure.__set__(self, value)

    @classmethod
    def _combine_options(cls, user_options: dict | None) -> dict:
        """Overlay ``user_options`` on :attr:`DEFAULT_OPTIONS`, validating each value."""
        options = cls._overlay_options(
            cls.DEFAULT_OPTIONS, user_options, solver_name="diffsol"
        )
        for name, default in cls._INTEGRATOR_DEFAULTS.items():
            options[name] = cls._checked_option(name, options[name], default)
        options["num_threads"] = cls._checked_count(
            "num_threads", options["num_threads"], minimum=1
        )
        return options

    @classmethod
    def _checked_option(cls, name: str, value, default):
        """Validate an integrator option as the type of its default."""
        if isinstance(default, int):
            return cls._checked_count(name, value, minimum=0)
        if value is None and default is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            kind = "a real number or None" if default is None else "a real number"
            raise pybamm.SolverError(f"{name} must be {kind}, got {value!r}")
        return float(value)

    @staticmethod
    def _checked_count(name: str, value, minimum: int) -> int:
        """Validate ``value`` as an integer from ``minimum`` to ``sys.maxsize``."""
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or not minimum <= value <= sys.maxsize
        ):
            raise pybamm.SolverError(
                f"{name} must be an integer from {minimum} to {sys.maxsize}, "
                f"got {value!r}"
            )
        return int(value)

    def _integrator_options(self) -> dict:
        """The subset of ``self._options`` diffsol's ``OdeSolverOptions`` takes."""
        return {key: self._options[key] for key in self._INTEGRATOR_DEFAULTS}

    @property
    def _internal_initialisation(self) -> bool:
        """Return True if using native diffsol IC calculation."""
        return self._calc_ic

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in _REBUILDABLE_STATE_KEYS:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Read as "not yet set up", so the next solve() rebuilds the prepared solver
        self._model_set_up = {}

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        """Set up the solver, building the Rust compiled model."""
        if model.convert_to_format != "rust":
            pybamm.logger.info(
                f"Converting {model.name} to Rust for solving with DiffsolSolver"
            )
            model.convert_to_format = "rust"

        if self._use_default_root_method and model.len_alg > 0:
            self.root_method = "nonlinear_solver"

        base_set_up_return = super().set_up(model, inputs, t_eval, ics_only)

        if ics_only:
            return base_set_up_return

        if isinstance(inputs, list):
            inputs_dict = inputs[0]
        else:
            inputs_dict = inputs or {}

        self._build_rust_model(model, inputs_dict)

        return base_set_up_return

    def _build_rust_model(self, model, inputs_dict):
        """Lower the model to a Rust ``CompiledModel`` and prepare the solver."""
        lowering = RustModelLowering(model, inputs_dict)
        lowering.state_residual()

        if model.calculate_sensitivities:
            validate_rust_sensitivity_widths(model, model.calculate_sensitivities)
            lowering.sensitivity_indices(model.calculate_sensitivities)

        # Time-integral outputs lower to their integrand; the sum runs in the assembly
        _, output_lens = lowering.outputs(
            self.output_variables, time_integral_vars=self._time_integral_vars
        )
        self._output_assembly = OutputAssembly.dense(
            self.output_variables,
            output_lens,
            time_integrals=self._time_integral_vars,
        )

        lowering.termination_events()

        self._rust_model = lowering.compile()
        lowering.bind_generic_evaluators(self._rust_model, root_method=self.root_method)
        if self._observes_via_compiled_model:
            ObserverCache.of(model).bind_rust_model(self._rust_model)

        from pybamm.rust import PreparedSolver

        try:
            self._prepared_solver = PreparedSolver(
                self._rust_model,
                float(self.rtol),
                self._check_atol_type(self.atol, model),
                self._sens_atol_factor,
                self._integrator_options(),
                # Outputs-only solves store no state trajectory to Hermite between.
                self._hermite_interpolation and not self.output_variables,
            )
        except ValueError as error:
            # The binding rejects an invalid tolerance or option with ValueError
            raise pybamm.SolverError(str(error)) from error

    def _integrate(
        self,
        model,
        t_eval,
        inputs_list=None,
        t_interp=None,
        nproc=1,
    ):
        """Integrate the model using the diffsol BDF solver.

        ``t_eval`` and ``t_interp`` merge into one output grid, with ``t_eval`` also
        the stop times. ``nproc`` is ignored: concurrency is the ``num_threads``
        option, as ``PreparedSolver`` cannot be pickled to worker processes.
        """
        if not hasattr(self, "_prepared_solver"):
            raise pybamm.SolverError("DiffsolSolver requires set_up() before solve()")

        inputs_list = inputs_list or [{}]
        t_solve = np.asarray(self._output_times(t_eval, t_interp), dtype=np.float64)
        t_stop = np.asarray(t_eval, dtype=np.float64)
        y0S_list = getattr(model, "y0S_list", None) or [None] * len(model.y0_list)

        n_sets = len(inputs_list)
        if self._options["num_threads"] > 1 and n_sets > 1:
            results = self._solve_batch(model, t_solve, t_stop, inputs_list, y0S_list)
        else:
            results = [
                self._solve_one(model, t_solve, t_stop, inputs_dict, y0, y0S)
                for inputs_dict, y0, y0S in zip(
                    inputs_list, model.y0_list, y0S_list, strict=True
                )
            ]

        failures = [
            f"input set {index}: {result}"
            for index, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise pybamm.SolverError("; ".join(failures))

        return [
            self._build_solution(model, inputs_dict, result)
            for inputs_dict, result in zip(inputs_list, results, strict=True)
        ]

    def _payload_flags(self, model) -> dict[str, bool]:
        """The ``outputs``/``sensitivities`` flags for ``PreparedSolver.solve``."""
        return {
            "outputs": bool(self.output_variables),
            "sensitivities": bool(model.calculate_sensitivities),
        }

    def process_t_interp(self, t_interp, t_eval=None) -> npt.NDArray[np.float64]:
        """Process ``t_interp``, defaulting to 100 points uniform over ``t_eval``.

        The default applies when ``t_interp`` is None, or empty with a two-point
        ``t_eval``.

        Parameters
        ----------
        t_interp : None, list or ndarray
            The requested interpolation times.
        t_eval : None, list or ndarray, optional
            The requested stop times.

        Returns
        -------
        ndarray
            The interpolation times.
        """
        requested = t_interp
        t_interp = super().process_t_interp(t_interp, t_eval=t_eval)
        if t_interp.size > 0 or t_eval is None or len(t_eval) < 2:
            return t_interp
        if requested is None or len(t_eval) == 2:
            # Distant output times would make off-grid reads one long chord or cubic
            return np.linspace(t_eval[0], t_eval[-1], 100)
        return t_interp

    @staticmethod
    def _output_times(t_eval, t_interp) -> npt.NDArray[np.float64]:
        """The grid the solver reports its dense output on over one segment."""
        if t_interp is None or len(t_interp) == 0:
            return np.asarray(t_eval, dtype=np.float64)
        t_interp = np.asarray(t_interp, dtype=np.float64)
        # A segment between discontinuities reports only its own share of t_interp
        inside = (t_interp >= t_eval[0]) & (t_interp <= t_eval[-1])
        return np.union1d(t_eval, t_interp[inside])

    def _solve_one(self, model, t_eval, t_stop, inputs_dict, y0, y0S=None):
        """Integrate one input set, returning the raw Rust result or its failure.

        ``t_stop`` entries not also in ``t_eval`` are integrated through.
        """
        y0_np = _as_flat_float64(y0)
        y0_sens = None
        if model.calculate_sensitivities and y0S is not None:
            y0_sens = _flatten_y0_sens(
                y0S, y0_np.size, len(model.calculate_sensitivities)
            )

        try:
            return self._prepared_solver.solve(
                t_eval,
                t_stop,
                y0_np,
                flatten_inputs(inputs_dict),
                y0_sens=y0_sens,
                num_threads=self._options["num_threads"],
                **self._payload_flags(model),
            )
        except RuntimeError as error:
            # Integration failures cross the FFI boundary as RuntimeError
            return error
        except ValueError as error:
            # An invalid request, such as sensitivities the model was compiled without
            raise pybamm.SolverError(str(error)) from error

    def _solve_batch(self, model, t_eval, t_stop, inputs_list, y0S_list):
        """Integrate every input set concurrently; failed sets return their error."""
        y0 = np.vstack([_as_flat_float64(y0) for y0 in model.y0_list])
        inputs = np.vstack([flatten_inputs(d) for d in inputs_list])
        y0_sens = None
        if model.calculate_sensitivities:
            y0_sens = self._stack_y0_sens(model, y0S_list, y0.shape[1])

        try:
            return self._prepared_solver.solve_batch(
                t_eval,
                t_stop,
                y0,
                inputs,
                # The configured width, so the process-wide pool cache is reused
                self._options["num_threads"],
                y0_sens=y0_sens,
                **self._payload_flags(model),
            )
        except ValueError as error:
            raise pybamm.SolverError(str(error)) from error

    @staticmethod
    def _stack_y0_sens(model, y0S_list, n_states):
        """The batch's ``dy0/dp`` seeds, one row per set, or ``None`` if all zero."""
        if not model.calculate_sensitivities:
            return None
        n_params = len(model.calculate_sensitivities)

        def seed(y0S):
            if y0S is None:
                return None
            flat = _flatten_y0_sens(y0S, n_states, n_params)
            return flat if flat.size else None

        rows = [seed(y0S) for y0S in y0S_list]
        if all(row is None for row in rows):
            return None
        zeros = np.zeros(n_states * n_params, dtype=np.float64)
        return np.vstack([zeros if row is None else row for row in rows])

    def _build_solution(self, model, inputs_dict, result):
        """Turn one raw Rust result into a :class:`pybamm.Solution`."""
        t = result.t
        if result.flag == 1 and t.size == 1:
            # Stopped at t0 on an event violated after diffsol's own initialisation
            self._check_event_violation(t, model, result.y_event, inputs_dict)
        termination = "event" if result.flag == 1 else "final time"
        t_event = None
        if result.t_event is not None:
            t_event = np.array([result.t_event])

        if self.output_variables:
            # The terminal full state, the only one outputs-only callers restart from
            y_event = np.asarray(result.y_event).reshape(-1, 1)
            sensitivity_names = model.calculate_sensitivities or []
            # The states have no sensitivities, as all_ys has no rows; the names stay
            state_sensitivities = {}
            if sensitivity_names:
                state_sensitivities = {
                    name: np.zeros((0, 1)) for name in sensitivity_names
                }
                state_sensitivities["all"] = np.zeros((0, len(sensitivity_names)))

            sol = pybamm.Solution(
                t,
                # Zero-row, not None: experiment-step stitching slices all_ys
                np.zeros((0, t.size)),
                model,
                inputs_dict,
                t_event,
                y_event,
                termination,
                all_sensitivities=state_sensitivities,
                variables_returned=True,
            )
            # first_state rebuilds from it, as all_ys holds no states
            sol._y0 = result.y_start
            sensitivities = None
            if sensitivity_names:
                # So first_state, last_state and a continued step have them
                sol._y0_sensitivities = _state_sensitivities(
                    np.vstack(result.yS_start), sensitivity_names
                )
                sol._y_event_sensitivities = _state_sensitivities(
                    np.vstack(result.yS_event), sensitivity_names
                )
                sensitivities = self._output_assembly.stack_parameter_blocks(
                    result.yS, t.size, sensitivity_names
                )
            # `result.y` holds output rows, output-major; the assembly wants time-major
            self._output_assembly.attach(
                sol,
                np.asarray(result.y).T,
                sensitivities=sensitivities,
                sensitivity_names=sensitivity_names,
            )
        else:
            y = result.y  # shape (n_states, n_times) from Rust

            y_event = None
            if t_event is not None and result.y_event is not None:
                y_event = np.asarray(result.y_event).reshape(-1, 1)

            yS_out = {}
            if model.calculate_sensitivities:
                sensitivity_names = model.calculate_sensitivities
                # Bind yS once — the getter rebuilds the list on every access.
                yS_list = result.yS
                yS_out = {
                    name: np.asarray(yS_list[i]).reshape(-1, 1)
                    for i, name in enumerate(sensitivity_names)
                }
                yS_out["all"] = np.hstack([yS_out[name] for name in sensitivity_names])

            sol = pybamm.Solution(
                t,
                y,
                model,
                inputs_dict,
                t_event,
                y_event,
                termination,
                all_sensitivities=yS_out,
                all_yps=result.yp,
            )

        # Names the event, so BaseSolver need not re-evaluate them all at y_event
        sol.closest_event_idx = result.event_index

        # Bind statistics once — the getter clones the struct on every access.
        statistics = result.statistics
        # Measured in Rust, as batched sets' wall clocks overlap
        sol.integration_time = statistics.integration_time_secs
        sol.solver_statistics = pybamm.SolverStatistics(
            number_of_steps=statistics.number_of_steps,
            number_of_linear_solver_setups=statistics.number_of_linear_solver_setups,
            number_of_nonlinear_solver_iterations=statistics.number_of_nonlinear_solver_iterations,
            number_of_nonlinear_solver_fails=statistics.number_of_nonlinear_solver_fails,
            number_of_error_test_failures=statistics.number_of_error_test_failures,
            number_of_linear_solver_setups_from_checkpoint=statistics.number_of_linear_solver_setups_from_checkpoint,
            number_of_linear_solver_setups_from_first_convergence_fail=statistics.number_of_linear_solver_setups_from_first_convergence_fail,
            number_of_linear_solver_setups_from_second_convergence_fail=statistics.number_of_linear_solver_setups_from_second_convergence_fail,
            number_of_linear_solver_setups_from_error_test_fail=statistics.number_of_linear_solver_setups_from_error_test_fail,
            number_of_linear_solver_setups_from_step_success=statistics.number_of_linear_solver_setups_from_step_success,
            problem_setup_time_secs=statistics.problem_setup_time_secs,
            ic_time_secs=statistics.ic_time_secs,
            solver_setup_time_secs=statistics.solver_setup_time_secs,
            sens_error_control_relaxed=statistics.sens_error_control_relaxed,
        )
        if sol.solver_statistics.sens_error_control_relaxed:
            pybamm.logger.warning(
                "The diffsol sensitivity solve failed under error control and was "
                "retried with sensitivities excluded from it, so their accuracy is "
                "not controlled. The likely cause is a `sens_atol_factor` too small "
                "for any step to meet; raise it if the gradients matter."
            )
        return sol
