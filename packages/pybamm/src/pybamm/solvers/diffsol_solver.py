"""Solver class wrapping the Rust diffsol BDF integrator."""

import numbers

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.solvers.base_solver import (
    flatten_inputs,
    validate_rust_sensitivity_widths,
)
from pybamm.solvers.observation import (
    NativeComputedObservation,
    NativeInterpolatingObservation,
    OutputAssembly,
)
from pybamm.solvers.rust_lowering import RustModelLowering


def _as_flat_float64(y0) -> npt.NDArray[np.float64]:
    """Flatten an initial state to a float64 vector (handles ``casadi.DM``)."""
    if hasattr(y0, "full"):
        y0 = y0.full()
    return np.asarray(y0, dtype=np.float64).flatten()


def _flatten_y0_sens(y0S, n_states: int, n_params: int) -> npt.NDArray[np.float64]:
    """Flatten ``dy0/dp`` to the column-major block the Rust solver expects.

    Every producer of ``model.y0S_list`` hands over one column per sensitivity
    parameter: ``(n_states, 1)`` from ``jacp``, bare ``(n_states,)`` from a step
    restart. A single ``(n_states, n_params)`` matrix is accepted too. All
    normalise to ``n_states * n_params`` values, parameter-outer/state-inner.

    Parameters
    ----------
    y0S : array-like or sequence of array-like
        Initial-condition sensitivities for one input set.
    n_states : int
        Number of states in the model.
    n_params : int
        Number of requested sensitivity parameters.

    Returns
    -------
    :class:`numpy.ndarray`
        Flat ``(n_states * n_params,)`` seed, or empty for an all-zero one.

    Raises
    ------
    :class:`pybamm.SolverError`
        If the flattened seed does not have ``n_states * n_params`` entries.
    """
    blocks = [y0S] if hasattr(y0S, "full") or isinstance(y0S, np.ndarray) else list(y0S)
    if not blocks:
        # Empty is the all-zero seed. A shape complaint here would mask the
        # solver's clearer error for a parameter it cannot differentiate.
        return np.zeros(0, dtype=np.float64)
    # column_stack lifts a bare (n_states,) column and leaves (n_states, 1) alone.
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

    This solver delegates time integration to a compiled Rust BDF (backward
    differentiation formula) implementation backed by the ``diffsol`` crate.
    The model's expression graph is lowered to Rust via ``to_rust`` and
    compiled into a ``CompiledModel`` that the Rust solver evaluates
    directly, avoiding per-step Python callbacks.

    The full state trajectory (and, by default, its time derivatives) is
    stored at every output time; the solvers differ only in which times those
    are. :class:`pybamm.IDAKLUSolver` uses its internal integrator steps as
    the output grid when ``t_interp`` is omitted, whereas diffsol evaluates
    its error-controlled dense output at the requested times alone. A bare
    ``t_eval=[t0, tf]`` span is answered on a uniform 100-point grid; any
    explicit output grid is honoured exactly.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float or :class:`numpy.ndarray`, optional
        The absolute tolerance for the solver, either shared by every state or
        one entry per state (default is 1e-6). A per-state array is the way to
        tolerance states of different magnitudes, since ``rtol`` already scales
        with each state's own value.
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
    root_tol : float, optional
        The tolerance for the initial-condition solver. Default is 1e-6.
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not.
    on_extrapolation : str, optional
        What to do if the solver is extrapolating. Options are "warn",
        "error", or "ignore". Default is "warn".
    on_failure : str, optional
        What to do if a solver error flag occurs. Options are "warn",
        "error", or "ignore". Default is "error".
    output_variables : list[str], optional
        List of variables to calculate and return. If none are specified
        then the complete state vector is returned (default is []).
    calc_ic : bool, optional
        If True, use native diffsol initial condition calculation instead of
        Python-side root-finding. Default is False.
    hermite_interpolation : bool, optional
        If True (default), also store the state time derivatives so off-grid
        ``sol[...](t)`` reads interpolate with cubic Hermite between output
        points, as :class:`pybamm.IDAKLUSolver` does. Disabling it halves
        trajectory memory; off-grid reads then interpolate linearly. Has no
        effect with ``output_variables``, whose solves store no state
        trajectory to interpolate. A solve asked for more than 4096 output
        points also drops the derivatives, the chord already sitting on the
        integration error floor by that density;
        ``Solution.hermite_interpolation`` reports which way a solve went.
    sens_atol_factor : float, optional
        Multiplier applied to ``atol`` on differential states to form the
        forward-sensitivity absolute tolerance floor (default 1e-3). Algebraic
        states keep ``atol``. Raise it towards 1.0 if a stiff DAE fails its
        sensitivity solve; lower it for tighter sensitivities at more steps.
    options : dict, optional
        Integrator tuning, one key per diffsol ``OdeSolverOptions`` knob, plus
        ``num_threads``: how many input sets solve concurrently. Unset keys take
        the defaults in :attr:`DEFAULT_OPTIONS`; an unknown key raises rather
        than being silently ignored. Leave ``num_threads`` at 1 under an
        outer-parallel caller (PyBOP, joblib, a ``ThreadPoolExecutor``), whose
        thread count would otherwise multiply with this one.
    """

    _integrates_via_compiled_model = True
    # Native (Rust) observation is the default for diffsol: the attach on the
    # full-state solve fires, and an un-lowerable variable hard-fails by design.
    _observes_via_compiled_model = True

    #: diffsol's own defaults, except the cumulative-per-solve
    #: ``max_nonlinear_solver_failures``: at diffsol's 50 it caps solve length
    #: rather than divergence, which ``min_timestep`` catches instead. Kept
    #: literal so importing ``pybamm`` never needs the extension; pinned
    #: against it by ``test_defaults_match_the_rust_defaults``.
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

    #: The integrator knobs plus ``num_threads``, which says how solves are
    #: executed rather than how one integrates and so never reaches diffsol.
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
        calc_ic=False,
        hermite_interpolation=True,
        sens_atol_factor=1e-3,
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
        self._options["num_threads"] = self._checked_num_threads(
            self._options["num_threads"]
        )

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

    @classmethod
    def _combine_options(cls, user_options: dict | None) -> dict:
        """Overlay ``user_options`` on :attr:`DEFAULT_OPTIONS`.

        Parameters
        ----------
        user_options : dict or None
            Overrides, keyed as diffsol's option names or ``num_threads``.

        Returns
        -------
        dict
            One entry per known option.

        Raises
        ------
        :class:`pybamm.SolverError`
            If a key is not a known option. The Rust side requires every
            integrator key, so a typo would otherwise be dropped silently.
        """
        return cls._overlay_options(
            cls.DEFAULT_OPTIONS, user_options, solver_name="diffsol"
        )

    @staticmethod
    def _checked_num_threads(num_threads) -> int:
        """Validate ``num_threads`` as a count of concurrent input sets.

        Parameters
        ----------
        num_threads : object
            The ``num_threads`` option as the caller supplied it.

        Returns
        -------
        int
            The validated count.

        Raises
        ------
        :class:`pybamm.SolverError`
            If it is not an integer of at least 1. Rust takes it as a ``usize``,
            which would reject a negative with a bare ``OverflowError``.
        """
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, numbers.Integral)
            or num_threads < 1
        ):
            raise pybamm.SolverError(
                f"num_threads must be an integer >= 1, got {num_threads!r}"
            )
        return int(num_threads)

    def _integrator_options(self) -> dict:
        """The subset of ``self._options`` diffsol's ``OdeSolverOptions`` takes."""
        return {key: self._options[key] for key in self._INTEGRATOR_DEFAULTS}

    @property
    def _internal_initialisation(self) -> bool:
        """Return True if using native diffsol IC calculation."""
        return self._calc_ic

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        """Set up the solver, building the Rust compiled model.

        Delegates to ``BaseSolver.set_up`` for initial-condition processing,
        then lowers the discretised model to a ``CompiledModel`` for the
        Rust diffsol backend.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        inputs : dict or list of dict, optional
            Any input parameters to pass to the model when solving.
        t_eval : numeric type, optional
            The times at which to stop the integration due to a
            discontinuity in time.
        ics_only : bool, optional
            If True, only process initial conditions (skip full setup).
        """
        if model.convert_to_format != "rust":
            pybamm.logger.info(
                f"Converting {model.name} to Rust for solving with DiffsolSolver"
            )
            model.convert_to_format = "rust"

        # Auto-select nonlinear_solver for DAE models if user didn't specify root_method
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
        """Lower the discretised PyBaMM model to a Rust ``CompiledModel``.

        Builds the expression graph, mass matrix, output variable
        expressions, and termination event expressions required by the
        Rust diffsol solver.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The discretised model.
        inputs_dict : dict
            Input parameter values.
        """
        lowering = RustModelLowering(model, inputs_dict)
        lowering.state_residual()

        if model.calculate_sensitivities:
            validate_rust_sensitivity_widths(model, model.calculate_sensitivities)
            lowering.sensitivity_indices(model.calculate_sensitivities)

        # A time-integral output lowers to its integrand trajectory; the postfix
        # sum runs post-solve, in the assembly.
        _, output_lens = lowering.outputs(
            self.output_variables, time_integral_vars=self._time_integral_vars
        )
        self._output_assembly = OutputAssembly(
            self.output_variables,
            output_lens,
            time_integrals=self._time_integral_vars,
        )

        lowering.termination_events()

        self._rust_model = lowering.compile()
        lowering.bind_generic_evaluators(self._rust_model)

        # Observation tapes cached 1:1 with the rust model so repeated solves
        # reuse them instead of recompiling against the retained graph.
        self._rust_observation_cache: dict = {}

        from pybamm.rust import PreparedSolver

        self._prepared_solver = PreparedSolver(
            self._rust_model,
            float(self.rtol),
            self._check_atol_type(self.atol, model),
            self._sens_atol_factor,
            self._integrator_options(),
            # Outputs-only solves store no state trajectory to Hermite between.
            self._hermite_interpolation and not self.output_variables,
        )

    def _set_consistent_initialization(self, model, time, inputs_list):
        super()._set_consistent_initialization(model, time, inputs_list)
        # first_state on an outputs-only solution rebuilds from y0full.
        model.y0full = [_as_flat_float64(y0) for y0 in model.y0_list]

    def _integrate(
        self,
        model,
        t_eval,
        inputs_list=None,
        t_interp=None,
        nproc=1,
    ):
        """Integrate the model using the diffsol BDF solver.

        Overrides the base class for two reasons. The diffsol backend uses dense
        output, so `t_eval` and `t_interp` merge into a single sorted array of
        output times, with `t_eval` also passed down as the stop times the
        integrator must land on and restart from. And concurrency over input
        sets is the solver's own, through the `num_threads` option and a rayon
        pool inside Rust: `nproc` is accepted for signature compatibility and
        ignored, because `PreparedSolver` is a PyO3 object that cannot be
        pickled to worker processes.
        """
        if not hasattr(self, "_prepared_solver"):
            raise RuntimeError("DiffsolSolver requires set_up() before solve()")

        inputs_list = inputs_list or [{}]
        # Shared by every set, so converted once rather than per solve.
        t_solve = np.asarray(self._output_times(t_eval, t_interp), dtype=np.float64)
        t_stop = np.asarray(t_eval, dtype=np.float64)
        # dy0/dp is per input set, like y0; absent when nothing is differentiated.
        y0S_list = getattr(model, "y0S_list", None) or [None] * len(model.y0_list)

        n_sets = len(inputs_list)
        if self._options["num_threads"] > 1 and n_sets > 1:
            results = self._solve_batch(model, t_solve, t_stop, inputs_list, y0S_list)
        else:
            results = [
                self._solve_one(model, t_solve, t_stop, i, inputs_dict, y0, y0S)
                for i, (inputs_dict, y0, y0S) in enumerate(
                    zip(inputs_list, model.y0_list, y0S_list, strict=True)
                )
            ]

        return [
            self._build_solution(model, inputs_dict, result)
            for inputs_dict, result in zip(inputs_list, results, strict=True)
        ]

    def _payload_flags(self, model) -> dict[str, bool]:
        """The payload the configured solve mode asks the Rust solver for.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model being solved, for whether sensitivities were requested.

        Returns
        -------
        dict
            ``outputs`` and ``sensitivities`` keyword arguments for
            :meth:`pybamm.rust.PreparedSolver.solve`.
        """
        return {
            "outputs": bool(self.output_variables),
            "sensitivities": bool(model.calculate_sensitivities),
        }

    @staticmethod
    def _output_times(t_eval, t_interp) -> npt.NDArray[np.float64]:
        """The grid the solver reports its dense output on.

        Parameters
        ----------
        t_eval : array-like
            Requested times, which are also the integrator's stop times.
        t_interp : array-like or None
            Extra interpolation times.

        Returns
        -------
        :class:`numpy.ndarray`
            Sorted output times.
        """
        if t_interp is not None and len(t_interp) > 0:
            return np.union1d(t_eval, t_interp)
        if len(t_eval) == 2:
            # A bare [t0, tf] span leaves the output times to the solver (IDAKLU
            # returns its steps); two points would interpolate as one chord.
            return np.linspace(t_eval[0], t_eval[-1], 100)
        return t_eval

    def _solve_one(self, model, t_eval, t_stop, index, inputs_dict, y0, y0S=None):
        """Integrate one input set, returning the raw Rust result.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution.
        t_stop : :class:`numpy.array`
            Discontinuity times the integrator must land on exactly and restart
            from. Every entry must also appear in ``t_eval``; those that do not
            are integrated through.
        index : int
            Position of this set in the sweep, named if it fails.
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving.
        y0 : array-like
            The initial conditions for the model.
        y0S : array-like or sequence of array-like, optional
            ``dy0/dp`` for this input set, one column per requested sensitivity
            parameter. ``None`` seeds the sensitivity system with zeros.

        Returns
        -------
        :class:`pybamm.rust.SolveOutcome`
            The solve's payloads, whichever the configured mode asked for.

        Raises
        ------
        :class:`pybamm.SolverError`
            If the integration fails.
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
                **self._payload_flags(model),
            )
        except RuntimeError as error:
            # Integration failures cross the FFI boundary as RuntimeError.
            self._raise_for_set(index, len(model.y0_list), inputs_dict, error)

    def _solve_batch(self, model, t_eval, t_stop, inputs_list, y0S_list):
        """Integrate every input set concurrently, returning the raw Rust results.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`
            Output times, shared by every set, as
            :meth:`_output_times` computed them.
        t_stop : :class:`numpy.array`
            Discontinuity times, shared by every set.
        inputs_list : list of dict
            One input dict per set, in the order the results are returned in.
        y0S_list : list
            One ``dy0/dp`` seed per set, entries ``None`` where absent.

        Returns
        -------
        list
            One :class:`pybamm.rust.SolveOutcome` per set, in input order.

        Raises
        ------
        :class:`pybamm.SolverError`
            If any set fails, naming which one and why.
        """
        y0 = np.vstack([_as_flat_float64(y0) for y0 in model.y0_list])
        inputs = np.vstack([flatten_inputs(d) for d in inputs_list])
        y0_sens = None
        if model.calculate_sensitivities:
            y0_sens = self._stack_y0_sens(model, y0S_list, y0.shape[1])

        results = self._prepared_solver.solve_batch(
            t_eval,
            t_stop,
            y0,
            inputs,
            # The configured width, not the sweep's: keying the process-wide pool
            # cache on the workload would build a fresh pool per distinct sweep size.
            self._options["num_threads"],
            y0_sens=y0_sens,
            **self._payload_flags(model),
        )

        # A failed set carries its exception rather than raising it, so the index
        # survives the crossing and the message can name the set.
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self._raise_for_set(i, len(results), inputs_list[i], result)
        return results

    @staticmethod
    def _raise_for_set(index, n_sets, inputs_dict, error):
        """Re-raise ``error`` as a :class:`pybamm.SolverError`.

        Names the input set when there is more than one, so a sweep's failure
        is attributable whether it ran batched or serially.

        Parameters
        ----------
        index : int
            Position of the failing set.
        n_sets : int
            Sets in the sweep.
        inputs_dict : dict
            The failing set's inputs.
        error : BaseException
            The underlying failure.

        Raises
        ------
        :class:`pybamm.SolverError`
            Always.
        """
        if n_sets > 1:
            raise pybamm.SolverError(
                f"input set {index} of {n_sets} ({inputs_dict}) failed: {error}"
            ) from error
        raise pybamm.SolverError(str(error)) from error

    @staticmethod
    def _stack_y0_sens(model, y0S_list, n_states):
        """The batch's ``dy0/dp`` seeds as one row per set, or ``None``.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Solved model, for the requested sensitivity parameters.
        y0S_list : list
            One seed per set, entries ``None`` where absent.
        n_states : int
            States in the model.

        Returns
        -------
        :class:`numpy.ndarray` or None
            ``(n_sets, n_states * n_params)`` seeds, or ``None`` when every set's
            seed is the all-zero one and Rust can default it.
        """
        if not model.calculate_sensitivities:
            return None
        n_params = len(model.calculate_sensitivities)

        # An empty seed and an all-zero one mean the same thing, so both
        # normalise to None here and the rectangular array is filled below.
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
        """Turn one raw Rust result into a :class:`pybamm.Solution`.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The solved model.
        inputs_dict : dict
            The input set this result came from.
        result : :class:`pybamm.rust.SolveOutcome`
            An outcome from :meth:`_solve_one` or :meth:`_solve_batch`.

        Returns
        -------
        :class:`pybamm.Solution`
            Solution object with times, states, and event data.
        """
        t = result.t
        termination = {0: "final time", 1: "event"}.get(result.flag, "failure")
        t_event = None
        if result.t_event is not None:
            t_event = np.array([result.t_event])

        if self.output_variables:
            # Always present on this path: the terminal (or root) full state,
            # the only state an outputs-only caller can restart from.
            y_event = np.asarray(result.y_event).reshape(-1, 1)

            sol = pybamm.Solution(
                t,
                # Zero-row, not None: experiment-step stitching slices all_ys
                # unconditionally.
                np.zeros((0, t.size)),
                model,
                inputs_dict,
                t_event,
                y_event,
                termination,
                variables_returned=True,
            )
            sensitivity_names = model.calculate_sensitivities or []
            sensitivities = None
            if sensitivity_names:
                # Bind yS once — the FFI getter rebuilds the list on every access.
                sensitivities = self._output_assembly.stack_parameter_blocks(
                    result.yS, t.size, sensitivity_names
                )
            # `result.y` holds output rows, not states, because the request asked
            # for them; Rust lays them out output-major, the assembly wants time.
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

        # Outputs-only too: first_state/last_state carry observable full states.
        if self._observes_via_compiled_model and model.convert_to_format == "rust":
            # Stored yps reroute observation to IDAKLU's yp-consuming path;
            # without them the solution is only read on its own grid.
            backend = (
                NativeInterpolatingObservation
                if sol.hermite_interpolation
                else NativeComputedObservation
            )
            sol.observation = backend.uniform(
                self._rust_model,
                len(sol.all_ys),
                cache=self._rust_observation_cache,
            )

        # Bind statistics once — the getter clones the struct on every access.
        statistics = result.statistics
        # Measured in Rust, not around this call: under a batch the wall clocks
        # overlap, so every set would be stamped with the batch duration.
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
            ic_time_secs=statistics.ic_time_secs,
            solver_setup_time_secs=statistics.solver_setup_time_secs,
            sens_error_control_relaxed=statistics.sens_error_control_relaxed,
        )
        if sol.solver_statistics.sens_error_control_relaxed:
            pybamm.logger.warning(
                "The diffsol sensitivity solve failed under error control and was "
                "retried with sensitivities excluded from it. Sensitivity accuracy "
                "may be reduced; try increasing `sens_atol_factor`."
            )
        return sol
