from __future__ import annotations

import copy
import itertools
import multiprocessing as mp
import numbers
import platform
import re
import warnings

import casadi
import numpy as np
import numpy.typing as npt

import pybamm
from pybamm import ParameterValues
from pybamm.expression_tree.binary_operators import _Heaviside
from pybamm.expression_tree.input_parameter import DUMMY_INPUT_PARAMETER_VALUE
from pybamm.models.base_model import ModelSolutionObservability

# CasADi function names allow only letters, digits and single underscores
_NOT_IDENTIFIER_CHARS = re.compile(r"[^a-z0-9]+")


class BaseSolver:
    """Solve a discretised model.

    Parameters
    ----------
    method : str, optional
        The method to use for integration, specific to each solver
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str or pybamm algebraic solver class, optional
        The method to use to find initial conditions (for DAE solvers).
        Default is "nonlinear_solver", which uses a custom Newton solver for consistent
        initial conditions (recommended). If "casadi", the solver uses
        casadi's Newton rootfinding algorithm as a fallback.
        Otherwise, the solver uses 'scipy.optimize.root' with method
        specified by 'root_method' (e.g. "lm", "hybr", ...)
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    extrap_tol : float, optional
        The tolerance to assert whether extrapolation occurs or not. Default is 0.
    output_variables : list[str], optional
        List of variables to calculate and return. If none are specified then
        the complete state vector is returned (can be very large) (default is [])
    on_extrapolation : str, optional
        What to do if the solver is extrapolating. Options are "warn", "error", or "ignore".
        Default is "warn".
    on_failure : str, optional
        What to do if a solver error flag occurs. Options are "warn", "error", or "ignore".
        Default is "error".
    store_first_last : bool, optional
        If True, only the first and last sample of each integration window are
        stored (one experiment step in :meth:`pybamm.Simulation.solve`, or the
        full ``[t_eval[0], t_eval[-1]]`` window in :meth:`solve`). Intended for
        memory-light long experiments whose post-processing only reads per-step
        first/last values. Only has effect on solvers that support intra-solve
        interpolation (e.g. :class:`pybamm.IDAKLUSolver`); a warning is issued
        otherwise. Note: when this flag is on, intra-step interpolation falls
        back to linear across the whole step, so it is **not** appropriate when
        post-processing queries a non-endpoint time within a step.
        Default is False.
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method=None,
        root_tol=1e-6,
        extrap_tol=None,
        on_extrapolation=None,
        on_failure=None,
        output_variables=None,
        store_first_last=False,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.root_tol = root_tol
        # Set ``on_failure`` before ``root_method`` so the setter can propagate
        # the parent's failure policy down to the constructed root solver.
        self.on_failure = on_failure or "error"
        self.root_method = root_method
        self.extrap_tol = extrap_tol or -1e-10
        self.output_variables = [] if output_variables is None else output_variables
        if not isinstance(store_first_last, bool):
            raise TypeError("store_first_last must be a bool")
        self.store_first_last = store_first_last
        self._warned_store_first_last_no_op = False
        self.on_extrapolation = on_extrapolation or "warn"
        self._model_set_up = {}

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self._ode_solver = False
        self._algebraic_solver = False
        self._supports_interp = False
        self._supports_t_eval_discontinuities = False
        self.computed_var_fcns = {}
        self._mp_context = self.get_platform_context(platform.system())

    def to_config(self) -> dict:
        """Convert this solver to a JSON-serialisable config dict.

        Returns a dict with a ``"type"`` key (the solver class name) and one
        key per serialisable ``__init__`` parameter.  ``CompositeSolver``
        instances include a ``"sub_solvers"`` key with nested config dicts.

        Returns
        -------
        dict
            Config dict suitable for passing back to :meth:`from_config`.
        """
        from pybamm.expression_tree.operations.serialise import Serialise

        return Serialise.serialise_solver(self)

    @staticmethod
    def from_config(data: dict) -> BaseSolver:
        """Create a solver from a config dict.

        Parameters
        ----------
        data : dict
            Config dict as produced by :meth:`to_config`, with a ``"type"``
            key and solver-specific keyword arguments.

        Returns
        -------
        :class:`BaseSolver`
        """
        from pybamm.expression_tree.operations.serialise import Serialise

        return Serialise.deserialise_solver(data)

    @property
    def ode_solver(self):
        return self._ode_solver

    @property
    def algebraic_solver(self):
        return self._algebraic_solver

    @property
    def supports_interp(self):
        return self._supports_interp

    @property
    def supports_t_eval_discontinuities(self):
        return self._supports_t_eval_discontinuities

    @property
    def on_extrapolation(self):
        return self._on_extrapolation

    @on_extrapolation.setter
    def on_extrapolation(self, value):
        if value not in ["warn", "error", "ignore"]:
            raise ValueError("on_extrapolation must be 'warn', 'error', or 'ignore'")
        self._on_extrapolation = value

    @property
    def on_failure(self):
        return self._on_failure

    @on_failure.setter
    def on_failure(self, value):
        if value not in ["warn", "error", "ignore"]:
            raise ValueError("on_failure must be 'warn', 'error', or 'ignore'")
        self._on_failure = value

        # Keep root solver's failure policy in sync with the parent's
        root_method = getattr(self, "_root_method", None)
        if isinstance(root_method, pybamm.BaseSolver):
            root_method.on_failure = value

    @property
    def root_method(self):
        return self._root_method

    @root_method.setter
    def root_method(self, method):
        if method == "nonlinear_solver":
            # use the tighter of the two tolerances; a per-state atol contributes
            # its tightest entry, since the root solver takes a single tolerance
            atol = min(self.root_tol, float(np.min(self.atol)))
            method = pybamm.NonlinearSolver(
                atol=atol, rtol=self.rtol, on_failure=self.on_failure
            )
        elif method == "casadi":
            warnings.warn(
                "root_method='casadi' is deprecated and will be removed in a future "
                "release. Use root_method='nonlinear_solver' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # CasadiAlgebraicSolver is itself deprecated; its warning is redundant
            # with the one above for this kwarg, so suppress it here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                method = pybamm.CasadiAlgebraicSolver(self.root_tol)
        elif isinstance(method, str):
            method = pybamm.AlgebraicSolver(method, self.root_tol)
        elif not (
            method is None
            or (
                isinstance(method, pybamm.BaseSolver)
                and method.algebraic_solver is True
            )
        ):
            raise pybamm.SolverError("Root method must be an algebraic solver")
        self._root_method = method

    def copy(self):
        """Returns a copy of the solver"""
        new_solver = copy.copy(self)
        # clear _model_set_up
        new_solver._model_set_up = {}
        return new_solver

    def set_up(
        self,
        model: pybamm.BaseModel,
        inputs: dict | list[dict] | None = None,
        t_eval=None,
        ics_only=False,
    ):
        """Unpack model, perform checks, and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict or list of dict, optional
            Any input parameters to pass to the model when solving
        t_eval : numeric type, optional
            Not used, kept for backwards compatibility.
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        inputs = inputs or [{}]

        if ics_only:
            pybamm.logger.info("Start solver set-up, initial_conditions only")
        else:
            pybamm.logger.info("Start solver set-up")

        self._check_and_prepare_model_inplace(model)

        # set default calculate sensitivities on model
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        self._set_up_model_sensitivities_inplace(model)

        vars_for_processing = self._get_vars_for_processing(model, inputs[0])

        # Process initial conditions
        initial_conditions, _, jacp_ic, _ = process(
            model.concatenated_initial_conditions,
            "initial_conditions",
            vars_for_processing,
            use_jacobian=False,
        )
        model.initial_conditions_eval = initial_conditions
        model.jacp_initial_conditions_eval = jacp_ic

        # set initial conditions
        self._set_initial_conditions(model, 0.0, inputs)

        if ics_only:
            pybamm.logger.info("Finish solver set-up")
            return

        # Process rhs, algebraic, residual and event expressions
        # and wrap in callables
        rhs, jac_rhs, jacp_rhs, jac_rhs_action = process(
            model.concatenated_rhs, "RHS", vars_for_processing
        )

        algebraic, jac_algebraic, jacp_algebraic, jac_algebraic_action = process(
            model.concatenated_algebraic, "algebraic", vars_for_processing
        )

        # combine rhs and algebraic functions
        if len(model.rhs) == 0:
            rhs_algebraic = model.concatenated_algebraic
        elif len(model.algebraic) == 0:
            rhs_algebraic = model.concatenated_rhs
        else:
            rhs_algebraic = pybamm.NumpyConcatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )

        (
            rhs_algebraic,
            jac_rhs_algebraic,
            jacp_rhs_algebraic,
            jac_rhs_algebraic_action,
        ) = process(rhs_algebraic, "rhs_algebraic", vars_for_processing)

        (
            casadi_switch_events,
            terminate_events,
            interpolant_extrapolation_events,
            t_discon_constant_symbols,
            discontinuity_events,
            discontinuity_periods,
        ) = self._set_up_events(model, inputs, vars_for_processing)

        # Add the solver attributes
        model.rhs_eval = rhs
        model.algebraic_eval = algebraic
        model.rhs_algebraic_eval = rhs_algebraic

        model.terminate_events_eval = terminate_events
        model.interpolant_extrapolation_events_eval = interpolant_extrapolation_events
        model.discontinuity_events_eval = discontinuity_events
        model.discontinuity_periods_eval = discontinuity_periods
        model.t_discon_constant_symbols = t_discon_constant_symbols

        model.jac_rhs_eval = jac_rhs
        model.jac_rhs_action_eval = jac_rhs_action
        model.jacp_rhs_eval = jacp_rhs

        model.jac_algebraic_eval = jac_algebraic
        model.jac_algebraic_action_eval = jac_algebraic_action
        model.jacp_algebraic_eval = jacp_algebraic

        model.jac_rhs_algebraic_eval = jac_rhs_algebraic
        model.jac_rhs_algebraic_action_eval = jac_rhs_algebraic_action
        model.jacp_rhs_algebraic_eval = jacp_rhs_algebraic

        # Save CasADi functions for the CasADi solver
        # Save CasADi functions for solvers that use CasADi
        # Note: when we pass to casadi the ode part of the problem must be in
        if isinstance(self.root_method, pybamm.CasadiAlgebraicSolver) or isinstance(
            self,
            pybamm.CasadiSolver | pybamm.CasadiAlgebraicSolver,
        ):
            # can use DAE solver to solve model with algebraic equations only
            if len(model.rhs) > 0:
                t_casadi = vars_for_processing["t_casadi"]
                y_casadi = vars_for_processing["y_casadi"]
                p_casadi_stacked = vars_for_processing["p_casadi_stacked"]
                rhs_eval = rhs(t_casadi, y_casadi, p_casadi_stacked)
                if model.is_standard_form_dae:
                    explicit_rhs = rhs_eval
                else:
                    M_ode = casadi.DM(
                        model.mass_matrix.entries[: model.len_rhs, : model.len_rhs]
                    )
                    explicit_rhs = casadi.solve(M_ode, rhs_eval)
                model.casadi_rhs = casadi.Function(
                    "rhs", [t_casadi, y_casadi, p_casadi_stacked], [explicit_rhs]
                )
            model.casadi_switch_events = casadi_switch_events
            model.casadi_algebraic = algebraic
            model.casadi_sensitivities = jacp_rhs_algebraic
            model.casadi_sensitivities_rhs = jacp_rhs
            model.casadi_sensitivities_algebraic = jacp_algebraic

        # if output_variables specified then convert functions to casadi
        # expressions for evaluation within the respective solver
        self.computed_var_fcns = {}
        self.computed_dvar_dy_fcns = {}
        self.computed_dvar_dp_fcns = {}
        self._time_integral_vars = {}
        function_names = BaseSolver._wrangle_names(self.output_variables)
        for key in self.output_variables:
            # Check for any ExplicitTimeIntegral or DiscreteTimeSum variables
            processed_time_integral = (
                pybamm.ProcessedVariableTimeIntegral.from_pybamm_var(
                    model.get_processed_variable_or_event(key),
                    model.len_rhs_and_alg,
                )
            )
            # We will evaluate the sum node in the solver and sum it afterwards
            if processed_time_integral is None:
                var = model.get_processed_variable_or_event(key)
            else:
                var = processed_time_integral.sum_node
                self._time_integral_vars[key] = processed_time_integral

            # Generate Casadi function to calculate variable and derivates
            # to enable sensitivites to be computed within the solver
            (
                self.computed_var_fcns[key],
                self.computed_dvar_dy_fcns[key],
                self.computed_dvar_dp_fcns[key],
                _,
            ) = process(
                var,
                function_names[key],
                vars_for_processing,
                use_jacobian=True,
                return_jacp_stacked=True,
            )

        pybamm.logger.info("Finish solver set-up")

    def _set_initial_conditions(self, model, time, inputs: list[dict]):
        # model should have been discretised or an error raised in Self._check_and_prepare_model_inplace
        len_tot = model.len_rhs_and_alg
        y_zero = np.zeros((len_tot, 1))

        stacked = model.uses_stacked_inputs
        model.y0_list = []
        model.y0S_list = [] if model.jacp_initial_conditions_eval is not None else None
        for ipts in inputs:
            if stacked:
                inputs_y0_ics = stack_inputs(ipts)
            else:
                inputs_y0_ics = ipts

            model.y0_list.append(
                model.initial_conditions_eval(time, y_zero, inputs_y0_ics)
            )

            if model.jacp_initial_conditions_eval is not None:
                if stacked:
                    inputs_jacp_ics = inputs_y0_ics
                else:
                    # we are calculating the derivative wrt the inputs
                    # so need to make sure we convert int -> float
                    # This is to satisfy JAX jacfwd function which requires
                    # float inputs
                    inputs_jacp_ics = {
                        key: float(value) if isinstance(value, int) else value
                        for key, value in ipts.items()
                    }

                model.y0S_list.append(
                    model.jacp_initial_conditions_eval(time, y_zero, inputs_jacp_ics)
                )

    @classmethod
    def _wrangle_name(cls, name: str) -> str:
        """
        Convert a variable name into a valid CasADi function name.

        Parameters
        ----------
        name : str
            The variable name.

        Returns
        -------
        str
            ``"v_"`` followed by the lower-cased name, with "%" spelled "prc" and
            each run of other characters outside ``[a-z0-9]`` replaced by one "_".
        """
        body = _NOT_IDENTIFIER_CHARS.sub("_", name.casefold().replace("%", "prc"))
        body = body.strip("_")
        return f"v_{body}" if body else "v"

    @classmethod
    def _wrangle_names(cls, names: list[str]) -> dict[str, str]:
        """
        Map variable names to distinct, valid CasADi function names.

        Parameters
        ----------
        names : list of str
            The variable names.

        Returns
        -------
        dict
            The function name for each variable name. Names that would collide,
            including with the ``_jac``, ``_jac_action`` and ``d..._dp``
            functions derived from them, get a numbered suffix.
        """

        def derived(base):
            return {base, f"{base}_jac", f"{base}_jac_action", f"d{base}_dp"}

        # Names must be distinct: the `compile` option links them into one library
        used = set()
        wrangled = {}
        for name in names:
            base = candidate = cls._wrangle_name(name)
            suffix = 0
            while derived(candidate) & used:
                suffix += 1
                candidate = f"{base}_{suffix}"
            used |= derived(candidate)
            wrangled[name] = candidate
        return wrangled

    def _check_and_prepare_model_inplace(self, model):
        """
        Performs checks on the model and prepares it for solving.
        """
        # Check model.algebraic for ode solvers
        if self.ode_solver is True and len(model.algebraic) > 0:
            raise pybamm.SolverError(
                f"Cannot use ODE solver '{self.name}' to solve DAE model"
            )
        # Check model.rhs for algebraic solvers
        if self._algebraic_solver is True and len(model.rhs) > 0:
            raise pybamm.SolverError(
                """Cannot use algebraic solver to solve model with time derivatives"""
            )
        # casadi solver won't allow solving algebraic model so we have to raise an
        # error here
        if isinstance(self, pybamm.CasadiSolver) and len(model.rhs) == 0:
            raise pybamm.SolverError(
                "Cannot use CasadiSolver to solve algebraic model, "
                "use CasadiAlgebraicSolver instead"
            )
        # Discretise model if it isn't already discretised
        # This only works with purely 0D models, as otherwise the mesh and spatial
        # method should be specified by the user
        if model.is_discretised is False:
            try:
                disc = pybamm.Discretisation()
                disc.process_model(model)
            except pybamm.DiscretisationError as e:
                raise pybamm.DiscretisationError(
                    "Cannot automatically discretise model, "
                    f"model should be discretised before solving ({e})"
                ) from e

        if (
            isinstance(self, pybamm.CasadiSolver | pybamm.CasadiAlgebraicSolver)
        ) and model.convert_to_format != "casadi":
            pybamm.logger.warning(
                f"Converting {model.name} to CasADi for solving with CasADi solver"
            )
            model.convert_to_format = "casadi"
        if (
            isinstance(self.root_method, pybamm.CasadiAlgebraicSolver)
            and model.convert_to_format != "casadi"
        ):
            pybamm.logger.warning(
                f"Converting {model.name} to CasADi for calculating ICs with CasADi"
            )
            model.convert_to_format = "casadi"

    @staticmethod
    def _get_vars_for_processing(model, inputs: dict):
        vars_for_processing = {
            "model": model,
        }

        if model.convert_to_format != "casadi":
            # Create Jacobian from concatenated rhs and algebraic
            y = pybamm.StateVector(slice(0, model.len_rhs_and_alg))
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
            vars_for_processing.update({"y": y, "jacobian": jacobian})
            return vars_for_processing

        else:
            # Convert model attributes to casadi
            t_casadi = casadi.MX.sym("t")
            # Create the symbolic state vectors
            y_diff = casadi.MX.sym("y_diff", model.len_rhs)
            y_alg = casadi.MX.sym("y_alg", model.len_alg)
            y_casadi = casadi.vertcat(y_diff, y_alg)
            p_casadi = {}
            for name, value in inputs.items():
                if isinstance(value, numbers.Number):
                    p_casadi[name] = casadi.MX.sym(name)
                else:
                    p_casadi[name] = casadi.MX.sym(name, value.shape[0])
            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])
            vars_for_processing.update(
                {
                    "t_casadi": t_casadi,
                    "y_diff": y_diff,
                    "y_alg": y_alg,
                    "y_casadi": y_casadi,
                    "p_casadi": p_casadi,
                    "p_casadi_stacked": p_casadi_stacked,
                }
            )

            return vars_for_processing

    @staticmethod
    def _set_up_model_sensitivities_inplace(model):
        """
        Set up model attributes related to sensitivities.
        """
        if model.mass_matrix is None:
            return

        model.mass_matrix = pybamm.Matrix(
            model.mass_matrix.entries[: model.len_rhs_and_alg, : model.len_rhs_and_alg]
        )

    def _set_up_events(self, model, inputs: list[dict], vars_for_processing):
        # Check for heaviside and modulo functions in rhs and algebraic and add
        # discontinuity events if these exist.
        # Note: only checks for the case of t < X, t <= X, X < t, or X <= t,
        # but also accounts for the fact that t might be dimensional

        def supports_t_eval_discontinuities(expr):
            # Only IDAKLUSolver supports discontinuities represented by t_eval
            return self.supports_t_eval_discontinuities and expr.is_constant()

        def process_time(expression, name):
            # Evaluated as the residual is, since numpy can round it differently
            return process(expression, name, vars_for_processing, use_jacobian=False)[0]

        def add_discontinuity_event(name, expression):
            # Set-up reruns on the same model, which may already have this event
            for event in model.events:
                if (
                    event.event_type == pybamm.EventType.DISCONTINUITY
                    and event.name == name
                    and event.expression == expression
                ):
                    return
            model.events.append(
                pybamm.Event(name, expression, pybamm.EventType.DISCONTINUITY)
            )

        # Find all the constant time-based discontinuities
        t_discon_symbols = []
        # Periods of `t mod period` that restart the solver, expanded at each solve,
        # with their evaluators
        discontinuity_periods = []

        for symbol in itertools.chain(
            model.concatenated_rhs.pre_order(),
            model.concatenated_algebraic.pre_order(),
        ):
            if isinstance(symbol, _Heaviside):
                if symbol.right == pybamm.t:
                    expr = symbol.left
                elif symbol.left == pybamm.t:
                    expr = symbol.right
                else:
                    # Heaviside function does not contain pybamm.t as an argument.
                    # Do not create an event
                    continue  # pragma: no cover

                if supports_t_eval_discontinuities(expr):
                    # save the symbol and expression ready for evaluation later
                    t_discon_symbols.append((symbol, expr))
                else:
                    add_discontinuity_event(str(symbol), expr)

            elif isinstance(symbol, pybamm.Modulo) and symbol.left == pybamm.t:
                expr = symbol.right
                self._check_modulo_period(expr)
                if supports_t_eval_discontinuities(expr):
                    t_discon_symbols.append((symbol, expr))
                else:
                    name = f"period_event_{len(discontinuity_periods)}"
                    discontinuity_periods.append((expr, process_time(expr, name)))
            else:
                continue

        casadi_switch_events = []
        terminate_events = []
        interpolant_extrapolation_events = []
        discontinuity_events = []
        for n, event in enumerate(model.events):
            if event.event_type == pybamm.EventType.DISCONTINUITY:
                # Evaluated before the solver is called, for each input set
                discontinuity_events.append(
                    process_time(event.expression, f"event_{n}")
                )
            elif event.event_type == pybamm.EventType.SWITCH and (
                isinstance(self, pybamm.CasadiSolver)
                and self.mode == "fast with events"
                and model.algebraic != {}
            ):
                # Save some events to casadi_switch_events for the 'fast with
                # events' mode of the casadi solver
                # We only need to do this if the model is a DAE model
                # see #1082
                k = 20
                # address numpy 1.25 deprecation warning: array should have
                # ndim=0 before conversion
                # note: assumes that the sign for all batches is the same
                init_sign = float(
                    np.sign(
                        event.evaluate(0, model.y0_list[0].full(), inputs=inputs[0])
                    ).item()
                )
                # We create a sigmoid for each event which will multiply the
                # rhs. Doing * 2 - 1 ensures that when the event is crossed,
                # the sigmoid is zero. Hence the rhs is zero and the solution
                # stays constant for the rest of the simulation period
                # We can then cut off the part after the event was crossed
                event_sigmoid = (
                    pybamm.sigmoid(0, init_sign * event.expression, k) * 2 - 1
                )
                event_casadi = process(
                    event_sigmoid,
                    f"event_{n}",
                    vars_for_processing,
                    use_jacobian=False,
                )[0]
                # use the actual casadi object as this will go into the rhs
                casadi_switch_events.append(event_casadi)
            else:
                # use the function call
                event_call = process(
                    event.expression,
                    f"event_{n}",
                    vars_for_processing,
                    use_jacobian=False,
                )[0]
                if event.event_type == pybamm.EventType.TERMINATION:
                    terminate_events.append(event_call)
                elif event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION:
                    interpolant_extrapolation_events.append(event_call)

        return (
            casadi_switch_events,
            terminate_events,
            interpolant_extrapolation_events,
            t_discon_symbols,
            discontinuity_events,
            discontinuity_periods,
        )

    @staticmethod
    def _check_modulo_period(period: pybamm.Symbol) -> None:
        """Raise a ``SolverError`` if ``period`` is not a valid period of time.

        Parameters
        ----------
        period : :class:`pybamm.Symbol`
            The period of a modulo of time.

        Raises
        ------
        :class:`pybamm.SolverError`
            If ``period`` is not a scalar, depends on time or the state, or is a
            constant that is not positive.
        """
        if period.size != 1 or period.has_symbol_of_classes(
            (pybamm.Time, pybamm.StateVectorBase)
        ):
            raise pybamm.SolverError(
                f"The period of '{period}' in a modulo of time must be a scalar that "
                "depends only on input parameters"
            )
        if period.is_constant():
            value = float(np.ravel(period.evaluate())[0])
            if not value > 0:
                raise pybamm.SolverError(
                    f"The period of '{period}' in a modulo of time must be positive, "
                    f"but is {value}"
                )

    def _set_consistent_initialization(
        self, model: pybamm.BaseModel, time: float, inputs_list: list[dict]
    ):
        """
        Set initialized states for the model. This is skipped if the solver is an
        algebraic solver (since this would make the algebraic solver redundant), and if
        the model doesn't have any algebraic equations (since there are no initial
        conditions to be calculated in this case).

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : numeric type
            The time at which to calculate the initial conditions
        inputs_list : list of dict
            Any input parameters to pass to the model when solving
        """

        if self._algebraic_solver or model.len_alg == 0:
            # Don't update model.y0_list
            return

        # Calculate consistent states for the algebraic equations
        model.y0_list = self.calculate_consistent_state(model, time, inputs_list)

    def calculate_consistent_state(
        self, model: pybamm.BaseModel, time: float = 0, inputs: list[dict] | None = None
    ):
        """
        Calculate consistent state for the algebraic equations through
        root-finding. model.y0_list is used as the initial guess for rootfinding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : float
            The time at which to calculate the initial conditions
        inputs: list of dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations). If self.root_method == None then returns
            model.y0_list.
        """
        pybamm.logger.debug("Start calculating consistent states")
        inputs = inputs or [{}]

        if self.root_method is None:
            return model.y0_list
        try:
            root_sols = self.root_method._integrate(model, np.array([time]), inputs)
        except pybamm.SolverError as e:
            raise pybamm.SolverError(
                f"Could not find consistent states: {e.args[0]}"
            ) from e
        pybamm.logger.debug("Found consistent states")

        y0s = []
        for s in root_sols:
            self.check_extrapolation(s, model.events)
            y0s.append(s.all_ys[0])
        return y0s

    @staticmethod
    def _solve_process_calculate_sensitivities_arg(
        inputs: dict,
        model: pybamm.BaseModel,
        calculate_sensitivities: list[str] | bool,
        supplied_inputs: dict | None,
    ) -> tuple[list[str], bool]:
        """
        Resolve ``calculate_sensitivities`` into sorted input names and store them
        on the model.

        Parameters
        ----------
        inputs : dict
            The model's input values, one per input parameter of the model.
        model : :class:`pybamm.BaseModel`
            The model being solved.
        calculate_sensitivities : list of str or bool
            ``True`` for every input of the model, ``False`` for none, or a list of
            input names.
        supplied_inputs : dict or None
            The inputs as given by the caller.

        Returns
        -------
        list of str
            The sorted names of the inputs to differentiate with respect to.
        bool
            Whether these names differ from the model's previous ones.

        Raises
        ------
        :class:`pybamm.SolverError`
            If a name is not an input parameter of the model, or is vector-valued.
        """
        excluded_sensitivity_inputs = {pybamm.Simulation._STEP_INDEX_INPUT}
        unused_inputs = []

        # get a list-only version of calculate_sensitivities
        if isinstance(calculate_sensitivities, bool):
            if calculate_sensitivities:
                calculate_sensitivities_list = [
                    p for p in inputs if p not in excluded_sensitivity_inputs
                ]
                supplied = set() if supplied_inputs is None else set(supplied_inputs)
                unused_inputs = sorted(
                    supplied - set(inputs) - excluded_sensitivity_inputs
                )
            else:
                calculate_sensitivities_list = []
        else:
            calculate_sensitivities_list = [
                p
                for p in calculate_sensitivities
                if p not in excluded_sensitivity_inputs
            ]
            unknown_inputs = [
                p for p in calculate_sensitivities_list if p not in inputs
            ]
            if unknown_inputs:
                model_inputs = sorted(set(inputs) - excluded_sensitivity_inputs)
                raise pybamm.SolverError(
                    f"Cannot calculate sensitivities with respect to {unknown_inputs}: "
                    f"the input parameters of model '{model.name}' are {model_inputs}."
                )

        vector_inputs = [
            p for p in calculate_sensitivities_list if np.size(inputs[p]) > 1
        ]
        if vector_inputs:
            raise pybamm.SolverError(
                "Cannot calculate sensitivities with respect to vector-valued "
                f"input(s) {vector_inputs}; only scalar inputs are supported."
            )

        calculate_sensitivities_list.sort()
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        # Check that calculate_sensitivites have not been updated
        sensitivities_have_changed = (
            calculate_sensitivities_list != model.calculate_sensitivities
        )
        # Log once per set-up, not on every repeated solve
        if unused_inputs and (
            sensitivities_have_changed or not calculate_sensitivities_list
        ):
            pybamm.logger.warning(
                f"Inputs {unused_inputs} are not input parameters of model "
                f"'{model.name}', so no sensitivities are calculated for them."
            )

        # save sensitivity parameters so we can identify them later on
        # (FYI: this is used in the Solution class)
        model.calculate_sensitivities = calculate_sensitivities_list

        return calculate_sensitivities_list, sensitivities_have_changed

    def _check_single_model(self, model: pybamm.BaseModel) -> None:
        """Raise a ``RuntimeError`` if this solver is set up for a different model."""
        if self._model_set_up and model not in self._model_set_up:
            existing_model = next(iter(self._model_set_up))
            raise RuntimeError(
                "This solver has already been initialised for model "
                f'"{existing_model.name}". Please create a separate '
                "solver for this model"
            )

    def _integrate(
        self,
        model: pybamm.BaseModel,
        t_eval,
        inputs_list: list[dict] | None = None,
        t_interp=None,
        nproc=1,
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_list : list of dict, optional
            Any input parameters to pass to the model when solving
        """

        inputs_list = inputs_list or [{}]

        if len(inputs_list) == 1:
            new_solutions = [
                self._integrate_single(model, t_eval, inputs_list[0], model.y0_list[0])
            ]
        elif nproc == 1:
            # A one-process pool runs the same solves in the same order as this
            # loop, plus a spawn and a pickle of the model per input set.
            new_solutions = [
                self._integrate_single(model, t_eval, inputs, y0)
                for inputs, y0 in zip(inputs_list, model.y0_list, strict=True)
            ]
        else:
            with mp.get_context(self._mp_context).Pool(processes=nproc) as p:
                model_list = [model] * len(inputs_list)
                t_eval_list = [t_eval] * len(inputs_list)
                y0_list = model.y0_list
                new_solutions = p.starmap(
                    self._integrate_single,
                    zip(
                        model_list,
                        t_eval_list,
                        inputs_list,
                        y0_list,
                        strict=True,
                    ),
                )
                p.close()
                p.join()

        return new_solutions

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        """
        Solve a single model instance with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving
        y0 : array-like
            The initial conditions for the model
        """
        raise NotImplementedError("BaseSolver does not implement _integrate_single.")

    def solve(
        self,
        model,
        t_eval=None,
        inputs: list[dict] | dict | None = None,
        nproc=None,
        calculate_sensitivities=False,
        t_interp=None,
    ):
        """
        Execute the solver setup and calculate the solution of the model at
        specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions. All calls to solve must pass in the same model or
            an error is raised
        t_eval : None, list or ndarray, optional
            The times (in seconds) at which to compute the solution. Defaults to None.
        inputs : dict or list of dict, optional
            A dictionary or list of dictionaries describing any input parameters to
            pass to the model when solving
        nproc : int, optional
            Number of processes to use when solving for more than one set of input
            parameters. Defaults to value returned by "os.cpu_count()".
        calculate_sensitivities : list of str or bool, optional
            Whether the solver calculates sensitivities with respect to input parameters.
            Defaults to False. If True, sensitivities are calculated for every input
            parameter of the model. A list must name input parameters of the model.
            Only scalar inputs are supported. The columns of ``sensitivities["all"]``
            follow ``Solution.sensitivity_names``, which is sorted. **Limitations**:
            sensitivities are not calculated up to numerical tolerances so are not
            guaranteed to be within the tolerances set by the solver, please raise an
            issue if you require this functionality. Also, when using this feature with
            `pybamm.Experiment`, the sensitivities do not take into account the movement
            of step-transitions wrt input parameters, so do not use this feature if the
            timings of your experimental protocol change rapidly with respect to your
            input parameters.
            Likewise, they do not include the movement of a discontinuity whose time
            depends on an input.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to None.
            Only valid for solvers that support intra-solve interpolation (`IDAKLUSolver`).
        Returns
        -------
        :class:`pybamm.Solution` or list of :class:`pybamm.Solution` objects.
             If type of `inputs` is `list`, return a list of corresponding
             :class:`pybamm.Solution` objects. If no float lies between the start of
             `t_eval` and a discontinuity whose time depends on an input, the solution
             starts just after the discontinuity.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}` and
            `model.variables = {}`)
        :class:`pybamm.SolverError`
            If `t_eval` or the period of a modulo of time is invalid, or the input sets
            have discontinuities at different times
        :class:`RuntimeError`
            If multiple calls to `solve` pass in different models

        """
        pybamm.logger.info(f"Start solving {model.name} with {self.name}")

        # Make sure model isn't empty
        self._check_empty_model(model)

        # t_eval can only be None if the solver is an algebraic solver. In that case
        # set it to 0
        if t_eval is None:
            if self._algebraic_solver is False:
                raise ValueError("t_eval cannot be None")
            t_eval = np.array([0])

        # If t_eval is provided as [t0, tf] return the solution at 100 points
        elif isinstance(t_eval, list):
            if len(t_eval) == 1 and self._algebraic_solver is True:
                t_eval = np.array(t_eval)
            elif len(t_eval) != 2:
                raise pybamm.SolverError(
                    "'t_eval' can be provided as an array of times at which to "
                    "return the solution, or as a list [t0, tf] where t0 is the "
                    "initial time and tf is the final time, but has been provided "
                    f"as a list of length {len(t_eval)}."
                )
            elif not self.supports_interp:
                t_eval = np.linspace(t_eval[0], t_eval[-1], 100)

        # Make sure t_eval is monotonic
        if (np.diff(t_eval) < 0).any():
            raise pybamm.SolverError("t_eval must increase monotonically")

        endpoints = self._store_first_last_endpoints(t_eval)
        if endpoints is not None:
            t_eval = endpoints

        t_interp = self.process_t_interp(t_interp, t_eval=t_eval)

        # Set up inputs
        if isinstance(inputs, (dict, pybamm.ParameterValues)):
            inputs_list = [inputs]
        else:
            inputs_list = inputs or [{}]

        model_inputs_list: list[dict] = [
            self._set_up_model_inputs(model, inputs) for inputs in inputs_list
        ]

        # Before the sensitivity setting below changes the model
        self._check_single_model(model)
        _, sensitivities_have_changed = (
            BaseSolver._solve_process_calculate_sensitivities_arg(
                model_inputs_list[0],
                model,
                calculate_sensitivities,
                supplied_inputs=inputs_list[0],
            )
        )

        # (Re-)calculate consistent initialization
        # if any setup configuration has changed, we need to re-set up
        if sensitivities_have_changed:
            self._model_set_up.pop(model, None)
            # CasadiSolver caches its integrators using model, so delete this too
            if isinstance(self, pybamm.CasadiSolver):
                self.integrators.pop(model, None)

        # Set up (if not done already)
        timer = pybamm.Timer()
        # Set the initial conditions
        if model not in self._model_set_up:
            self.set_up(model, model_inputs_list, t_eval)
            self._model_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )
        elif (
            self._model_set_up[model]["initial conditions"]
            != model.concatenated_initial_conditions
        ):
            if self._algebraic_solver:
                # For an algebraic solver, we don't need to set up the initial
                # conditions function and we can just evaluate
                # model.concatenated_initial_conditions
                model.y0_list = [
                    model.concatenated_initial_conditions.evaluate()
                ] * len(inputs_list)
            else:
                # If the new initial conditions are different
                # and cannot be evaluated directly, set up again
                self.set_up(model, model_inputs_list, t_eval, ics_only=True)
            self._model_set_up[model]["initial conditions"] = (
                model.concatenated_initial_conditions
            )
        else:
            # Set the standard initial conditions
            self._set_initial_conditions(model, t_eval[0], model_inputs_list)

        # Process discontinuities, which can move the start to just after one
        t_eval_info = [
            self._get_discontinuity_start_end_indices(
                model, model.y0_list[i], inputs, t_eval
            )
            for i, inputs in enumerate(model_inputs_list)
        ]

        first_row = t_eval_info[0]
        for row in t_eval_info[1:]:
            if not all(
                np.array_equal(row_ele, first_row_ele)
                for row_ele, first_row_ele in zip(row, first_row, strict=True)
            ):
                # Can't handle different `t_eval`s for each input set
                raise pybamm.SolverError(
                    "Discontinuity events occur at different times between input parameter sets. "
                    "Please ensure that all input sets produce the same discontinuities."
                )

        t_start = t_eval[0]
        start_indices, end_indices, t_eval = first_row
        if t_eval[0] != t_start:
            t_interp = np.asarray(t_interp, dtype=float)
            t_interp = np.where(t_interp == t_start, t_eval[0], t_interp)

        # Solve for the consistent initialization
        self._set_consistent_initialization(model, t_eval[0], model_inputs_list)

        set_up_time = timer.time()
        timer.reset()

        # Check initial conditions don't violate events
        for y0, inpts in zip(model.y0_list, model_inputs_list, strict=True):
            self._check_event_violation_on_initialisation(t_eval, model, y0, inpts)

        solutions = self._integrate_segments(
            model,
            t_eval,
            start_indices,
            end_indices,
            model_inputs_list,
            t_interp,
            timer,
            nproc=nproc,
        )
        solve_time = timer.time()

        for i, solution in enumerate(solutions):
            # Check if extrapolation occurred
            self.check_extrapolation(solution, model.events)
            # Identify the event that caused termination and update the solution to
            # include the event time and state
            solutions[i], termination = self.get_termination_reason(
                solution, model.events
            )
            # Assign times
            solutions[i].set_up_time = set_up_time
            # all solutions get the same solve time, but their integration time
            # will be different (see https://github.com/pybamm-team/PyBaMM/pull/1261)
            solutions[i].solve_time = solve_time

        # Report times
        if len(solutions) == 1:
            pybamm.logger.info(f"Finish solving {model.name} ({termination})")
            pybamm.logger.info(
                f"Set-up time: {solutions[0].set_up_time}, Solve time: {solutions[0].solve_time} (of which integration time: {solutions[0].integration_time}), "
                f"Total time: {solutions[0].total_time}"
            )
        else:
            pybamm.logger.info(f"Finish solving {model.name} for all inputs")
            pybamm.logger.info(
                f"Set-up time: {solutions[0].set_up_time}, Solve time: {solutions[0].solve_time}, Total time: {solutions[0].total_time}"
            )

        # Raise error if solutions[0] only contains one timestep (except for algebraic
        # solvers, where we may only expect one time in the solution, or failure
        # solutions where we may only have the initial state)
        if (
            self._algebraic_solver is False
            and solutions[0].termination != "failure"
            and len(solutions[0].all_ts) == 1
            and len(solutions[0].all_ts[0]) == 1
        ):
            raise pybamm.SolverError(
                "Solution time vector has length 1. "
                "Check whether simulation terminated too early."
            )

        # Return solution(s)
        if len(solutions) == 1:
            return solutions[0]
        else:
            return solutions

    def _integrate_segments(
        self,
        model: pybamm.BaseModel,
        t_eval: npt.NDArray[np.float64],
        start_indices: list[int],
        end_indices: list[int],
        inputs_list: list[dict],
        t_interp: npt.NDArray[np.float64],
        timer: pybamm.Timer,
        nproc: int | None = None,
    ) -> list[pybamm.Solution]:
        """Integrate over each segment of ``t_eval`` between discontinuities.

        Each input set that reaches the end of a segment restarts from its final
        state and state sensitivities, made consistent if the model is a DAE. The
        initial states and their sensitivities are restored afterwards.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model to integrate, with ``y0_list`` set for each input set.
        t_eval : :class:`numpy.ndarray`
            The times of all segments.
        start_indices, end_indices : list of int
            The slice of ``t_eval`` that each segment spans.
        inputs_list : list of dict
            The input parameters of each input set.
        t_interp : :class:`numpy.ndarray`
            The interpolation times of all segments.
        timer : :class:`pybamm.Timer`
            The timer that gives each segment its solve time.
        nproc : int, optional
            Number of processes to use for more than one input set.

        Returns
        -------
        list of :class:`pybamm.Solution`
            The solution of each input set, up to the end of ``t_eval`` or the
            event that stopped it.
        """
        old_y0_list = model.y0_list
        old_y0S_list = model.y0S_list
        solutions = [None] * len(inputs_list)
        # Indices of the input sets that reached the end of every segment so far
        running = list(range(len(inputs_list)))
        for start_index, end_index in zip(start_indices, end_indices, strict=True):
            segment_t_eval = t_eval[start_index:end_index]
            pybamm.logger.verbose(
                f"Calling solver for {segment_t_eval[0]} < t < {segment_t_eval[-1]}"
            )
            new_solutions = self._integrate(
                model,
                segment_t_eval,
                [inputs_list[i] for i in running],
                self._segment_t_interp(
                    t_interp,
                    segment_t_eval,
                    first=start_index == 0,
                    last=end_index == len(t_eval),
                ),
                nproc=nproc,
            )
            # pybamm.Solution.__add__ assumes attribute solve_time
            solve_time = timer.time()
            for i, new_solution in zip(running, new_solutions, strict=True):
                new_solution.solve_time = solve_time
                if solutions[i] is None:
                    solutions[i] = new_solution
                else:
                    solutions[i] = solutions[i] + new_solution

            # A set that stopped on an event or failed is not restarted
            restarts = [
                (i, sol)
                for i, sol in zip(running, new_solutions, strict=True)
                if sol.termination == "final time"
            ]
            if not restarts or end_index == len(t_eval):
                break

            running = [i for i, _ in restarts]
            model.y0_list = [sol.last_state.all_ys[0][:, 0] for _, sol in restarts]
            if model.calculate_sensitivities:
                model.y0S_list = [
                    tuple(
                        self._restart_state_sensitivities(
                            sol, model.calculate_sensitivities
                        ).T
                    )
                    for _, sol in restarts
                ]
            self._set_consistent_initialization(
                model, t_eval[end_index], [inputs_list[i] for i in running]
            )

        model.y0_list = old_y0_list
        model.y0S_list = old_y0S_list
        return solutions

    @staticmethod
    def filter_discontinuities(t_discon: list, t_eval: list) -> np.ndarray:
        """
        Filter the discontinuities to only include the unique and sorted
        values within the t_eval range (non-exclusive of end points).

        Parameters
        ----------
        t_discon : list
            The list of all possible discontinuity times.
        t_eval : list
            The integration time points.

        Returns
        -------
        np.ndarray
            The filtered list of discontinuities within the range of t_eval.
        """
        t_discon_unique = np.unique(t_discon)

        # Find the indices within t_eval (non-exclusive of end points)
        idx_start = np.searchsorted(t_discon_unique, t_eval[0], side="right")
        idx_end = np.searchsorted(t_discon_unique, t_eval[-1], side="left")
        return t_discon_unique[idx_start:idx_end]

    def get_root_solver(self, model, inputs_dict, t_eval):
        if not model.algebraic_root_solver:
            model.algebraic_root_solver = self._set_up_root_solver(
                model, inputs_dict, t_eval
            )
        return model.algebraic_root_solver

    def _set_up_root_solver(self, model, inputs_dict, t_eval):
        raise NotImplementedError

    def _get_discontinuity_start_end_indices(self, model, y0, inputs, t_eval):
        # Copy as float: integer times would truncate the points inserted around
        # each discontinuity, and the caller's array must not be modified
        t_eval = np.array(t_eval, dtype=float)
        if self.supports_t_eval_discontinuities and model.t_discon_constant_symbols:
            pybamm.logger.verbose("Discontinuity events found for constant symbols")
            _t_discon_constant = []
            for symbol, expr in model.t_discon_constant_symbols:
                _t_discon_constant.extend(symbol._t_discon(expr, y0, inputs, t_eval))

            t_discon_constant = self.filter_discontinuities(_t_discon_constant, t_eval)
            t_eval = np.union1d(t_eval, t_discon_constant)

        discontinuity_periods = model.discontinuity_periods_eval
        if not model.discontinuity_events_eval and not discontinuity_periods:
            pybamm.logger.verbose("No additional discontinuity events found")
            return [0], [len(t_eval)], t_eval

        stacked_inputs = stack_inputs(inputs) if model.uses_stacked_inputs else None

        def evaluate(function):
            if stacked_inputs is None:
                return np.ravel(function(t=0, y=y0, inputs=inputs))
            # Much faster than numpy's conversion of a CasADi matrix
            return np.array(function(0, y0, stacked_inputs).elements())

        # The last float before and the first float after each discontinuity
        stops, restarts = [], []
        for event in model.discontinuity_events_eval:
            t_discon = evaluate(event)
            # Either side of a switch time may be continuous, so skip it
            stops.append(np.nextafter(t_discon, -np.inf))
            restarts.append(np.nextafter(t_discon, np.inf))
        for period_expression, period_eval in discontinuity_periods:
            period = evaluate(period_eval)[0]
            if not period > 0:
                raise pybamm.SolverError(
                    f"The period of '{period_expression}' in a modulo of time must "
                    f"be positive, but is {period} for inputs {inputs}"
                )
            period_stops, period_restarts = pybamm.Modulo._t_stops_restarts(
                period, t_eval
            )
            stops.append(period_stops)
            restarts.append(period_restarts)
        stops, restarts = np.concatenate(stops), np.concatenate(restarts)
        # Sorted by stop then restart, so that no restart precedes an earlier one
        order = np.lexsort((restarts, stops))
        stops, restarts = stops[order], restarts[order]
        # Only those with a time to stop at in t_eval[0] <= t < t_eval[-1]
        inside = (stops >= t_eval[0]) & (stops < t_eval[-1])
        stops, restarts = stops[inside], restarts[inside]

        pybamm.logger.verbose(f"Discontinuity events found at t = {restarts}")

        # insert time points around discontinuities in t_eval
        # keep track of subsections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        for before, after in zip(stops, restarts, strict=True):
            if after >= t_eval[-1]:
                # No time left to restart in, so integrate across it to the end
                break
            # Stop and restart in place of any time between them
            upper = np.searchsorted(t_eval, after, side="right")
            segment_start = start_indices[-1]
            if before > t_eval[segment_start]:
                lower = np.searchsorted(t_eval, before, side="left")
                t_eval = np.concatenate(
                    [t_eval[:lower], [before, after], t_eval[upper:]]
                )
                end_indices.append(lower + 1)
                start_indices.append(lower + 1)
            else:
                # No time to stop in since the segment started, so start it after
                t_eval = np.concatenate(
                    [t_eval[:segment_start], [after], t_eval[upper:]]
                )
        end_indices.append(len(t_eval))

        return start_indices, end_indices, t_eval

    def _check_event_violation_on_initialisation(self, *args, **kwargs):
        self._check_event_violation(*args, **kwargs)

    def _check_event_violation_post_solve(self, *args, **kwargs):
        self._check_event_violation(*args, **kwargs)

    def _check_event_violation(self, t_eval, model, y0, inputs_dict):
        num_terminate_events = len(model.terminate_events_eval)
        if num_terminate_events == 0:
            return

        inputs = inputs_dict
        if model.uses_stacked_inputs:
            inputs = stack_inputs(inputs_dict)

        events_eval = np.empty(num_terminate_events)
        for idx, event in enumerate(model.terminate_events_eval):
            if model.uses_stacked_inputs:
                event_eval = event(t_eval[0], y0, inputs)
            else:
                event_eval = event(t=t_eval[0], y=y0, inputs=inputs_dict)
            if not isinstance(event_eval, float):
                event_eval = np.asarray(event_eval).item()
            events_eval[idx] = event_eval

        if events_eval.min() <= 0:
            # find the events that were triggered by initial conditions
            termination_events = [
                x for x in model.events if x.event_type == pybamm.EventType.TERMINATION
            ]
            idxs = np.where(events_eval <= 0)[0]
            event_names = [termination_events[idx].name for idx in idxs]
            raise pybamm.SolverError(
                f"Events {event_names} are non-positive at initial conditions with inputs {inputs_dict}"
            )

    @staticmethod
    def _overlay_options(
        defaults: dict, user_options: dict | None, *, solver_name: str
    ) -> dict:
        """Overlay ``user_options`` on ``defaults``, rejecting unknown keys.

        Parameters
        ----------
        defaults : dict
            One entry per known option.
        user_options : dict or None
            Caller overrides.
        solver_name : str
            Solver named in the error message.

        Returns
        -------
        dict
            ``defaults`` with ``user_options`` applied.

        Raises
        ------
        :class:`pybamm.SolverError`
            If ``user_options`` has a key that ``defaults`` does not.
        """
        user_options = user_options or {}
        unknown = sorted(set(user_options) - set(defaults))
        if unknown:
            raise pybamm.SolverError(
                f"Unknown {solver_name} solver option(s): {', '.join(unknown)}. "
                f"Known options are: {', '.join(sorted(defaults))}."
            )
        return defaults | user_options

    @staticmethod
    def _check_restart_sensitivities(
        old_solution: pybamm.Solution, sensitivity_names: list[str]
    ) -> None:
        """Reject a sensitivity restart from a solution that cannot seed it.

        Parameters
        ----------
        old_solution : :class:`pybamm.Solution`
            The solution the next step would restart from.
        sensitivity_names : list of str
            The inputs the next step calculates sensitivities for.

        Raises
        ------
        :class:`pybamm.SolverError`
            If ``old_solution`` returned output variables only, or its final
            state has no sensitivities for some of ``sensitivity_names``.
        """
        if isinstance(old_solution, pybamm.EmptySolution):
            return
        # Not gated on _all_sensitivities: IDAKLU populates it at output width,
        # which cannot seed a state-width dy0/dp.
        if (
            old_solution.variables_returned
            and old_solution._y_event_sensitivities is None
        ):
            raise pybamm.SolverError(
                "Cannot continue a sensitivity solve from a solution that "
                "returned output variables only: the step boundary has no "
                "state sensitivities to seed dy0/dp from. Drop "
                "'output_variables' or 'calculate_sensitivities' for "
                "multi-step solves."
            )
        # Only the final segment's sensitivities describe the final state
        available = old_solution.sub_solutions[-1].sensitivity_names
        missing = [name for name in sensitivity_names if name not in available]
        if missing:
            raise pybamm.SolverError(
                f"The solution to continue from has no sensitivities for {missing}. "
                "Solve it with 'calculate_sensitivities' including these inputs."
            )

    @staticmethod
    def _check_saved_sensitivities(
        old_solution: pybamm.Solution, sensitivity_names: list[str]
    ) -> None:
        """Reject saving a step onto a solution with sensitivities for other inputs.

        Parameters
        ----------
        old_solution : :class:`pybamm.Solution`
            The solution the step would be appended to.
        sensitivity_names : list of str
            The inputs the step calculates sensitivities for.

        Raises
        ------
        :class:`pybamm.SolverError`
            If ``old_solution`` has sensitivities for a different set of inputs
            than ``sensitivity_names``.
        """
        if isinstance(old_solution, pybamm.EmptySolution):
            return
        old_names = sorted(old_solution.sensitivity_names)
        if old_names != sorted(sensitivity_names):
            raise pybamm.SolverError(
                f"Cannot save a step with sensitivities for {sensitivity_names} "
                f"onto a solution with sensitivities for {old_names}. Step with "
                "'save=False', or with the same 'calculate_sensitivities'."
            )

    @staticmethod
    def _restart_state_sensitivities(
        solution: pybamm.Solution, sensitivity_names: list[str]
    ) -> npt.NDArray[np.float64]:
        """State sensitivities at the final state of a solution, selected by name.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution to restart from.
        sensitivity_names : list of str
            The inputs to select, in the order of the returned columns.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of shape ``(number of states, len(sensitivity_names))``.
        """
        sensitivities = solution.last_state._all_sensitivities
        return np.hstack(
            [
                np.asarray(sensitivities[name][0], dtype=float).reshape(-1, 1)
                for name in sensitivity_names
            ]
        )

    def _set_sens_initial_conditions_from(
        self,
        solution: pybamm.Solution,
        model: pybamm.BaseModel,
        inputs: dict,
    ) -> tuple:
        """Sensitivities of the initial state set by name from a solution.

        Each state is stored as ``(var - reference) / scale``, where the
        reference and scale may depend on the inputs.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution to use to initialize the model.
        model : :class:`pybamm.BaseModel`
            The discretised model, with ``y0_list`` already set from ``solution``.
        inputs : dict
            The input parameters of ``model``.

        Returns
        -------
        tuple of :class:`numpy.ndarray`
            The initial state sensitivities, one array per name in
            ``model.calculate_sensitivities``.
        """
        names = model.calculate_sensitivities
        last_state = solution.last_state
        y0 = casadi.DM(model.y0_list[0]).full().ravel()
        p_casadi = {
            name: casadi.MX.sym(name, np.size(value)) for name, value in inputs.items()
        }
        p_casadi_stacked = casadi.vertcat(*p_casadi.values())
        p_sensitivities = casadi.vertcat(*[p_casadi[name] for name in names])
        p_stacked = flatten_inputs(inputs)

        y0S = np.zeros((model.len_rhs_and_alg, len(names)))
        for var in model.initial_conditions:
            sensitivities = last_state[var.name].sensitivities
            dvar_dp = np.hstack([sensitivities[name].reshape(-1, 1) for name in names])
            reference = var.reference.to_casadi(inputs=p_casadi)
            scale = var.scale.to_casadi(inputs=p_casadi)
            reference_and_scale = casadi.Function(
                "reference_and_scale",
                [p_casadi_stacked],
                [
                    scale,
                    casadi.jacobian(reference, p_sensitivities),
                    casadi.jacobian(scale, p_sensitivities),
                ],
            )
            scale_value, dreference_dp, dscale_dp = (
                value.full() for value in reference_and_scale(p_stacked)
            )
            # The derivative of (var - reference) / scale
            y_slice = model.y_slices[var][0]
            y0S[y_slice] = (
                dvar_dp - dreference_dp - y0[y_slice, np.newaxis] * dscale_dp
            ) / scale_value
        return tuple(y0S.T)

    def _store_first_last_endpoints(self, t_eval):
        if not self.store_first_last:
            return None
        if not self.supports_interp:
            if not self._warned_store_first_last_no_op:
                warnings.warn(
                    f"store_first_last has no effect on {self.name} "
                    "(no support for intra-solve interpolation); ignoring.",
                    pybamm.SolverWarning,
                    stacklevel=3,
                )
                self._warned_store_first_last_no_op = True
            return None
        if t_eval is None or len(t_eval) < 2:
            return None
        return np.array([t_eval[0], t_eval[-1]], dtype=float)

    def process_t_interp(self, t_interp, t_eval=None):
        endpoints = self._store_first_last_endpoints(t_eval)
        if endpoints is not None:
            return endpoints

        # set a variable for this
        no_interp = (not self.supports_interp) and (
            t_interp is not None and len(t_interp) != 0
        )
        if no_interp:
            warnings.warn(
                f"Explicit interpolation times not implemented for {self.name}",
                pybamm.SolverWarning,
                stacklevel=2,
            )

        if no_interp or t_interp is None:
            t_interp = np.empty(0)

        return t_interp

    @staticmethod
    def _segment_t_interp(
        t_interp, segment_t_eval: npt.NDArray[np.float64], first: bool, last: bool
    ) -> npt.NDArray[np.float64]:
        """Interpolation times of a solve that fall within one of its segments.

        Parameters
        ----------
        t_interp : array-like
            The interpolation times of the whole solve.
        segment_t_eval : :class:`numpy.ndarray`
            The times of the segment.
        first : bool
            Whether the segment starts the solve, so keeps the times before it.
        last : bool
            Whether the segment ends the solve, so keeps the times after it.

        Returns
        -------
        :class:`numpy.ndarray`
            ``t_interp`` if it is empty or the segment spans the whole solve,
            otherwise the times in the segment with its end points.
        """
        t_interp = np.asarray(t_interp, dtype=float)
        if t_interp.size == 0 or (first and last):
            return t_interp
        lower = -np.inf if first else segment_t_eval[0]
        upper = np.inf if last else segment_t_eval[-1]
        inside = t_interp[(t_interp >= lower) & (t_interp <= upper)]
        # An empty t_interp would store every step, so keep the end points
        return np.union1d(inside, segment_t_eval[[0, -1]])

    def step(
        self,
        old_solution,
        model,
        dt,
        t_eval=None,
        npts=None,
        inputs=None,
        save=True,
        calculate_sensitivities=False,
        t_interp=None,
        state_mapper=None,
    ):
        """
        Step the solution of the model forward by a given time increment. The
        first time this method is called it executes the necessary setup by
        calling `self.set_up(model)`.

        Parameters
        ----------
        old_solution : :class:`pybamm.Solution` or None
            The previous solution to be added to. If `None`, a new solution is created.
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        dt : numeric type
            The timestep (in seconds) over which to step the solution
        t_eval : list or numpy.ndarray, optional
            An array of times at which to stop the simulation and return the solution during the step
            (Note: t_eval is the time measured from the start of the step, so should start at 0 and end at dt).
            By default, the solution is returned at t0 and t0 + dt. If no float lies
            between the start of the step and a discontinuity whose time depends on an
            input, the step starts just after the discontinuity.
        npts : deprecated
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool, optional
            Save solution with all previous timesteps. Defaults to True.
        calculate_sensitivities : list of str or bool, optional
            Whether the solver calculates sensitivities with respect to input parameters.
            Defaults to False. If True, sensitivities are calculated for every input
            parameter of the model. A list must name input parameters of the model.
            Only scalar inputs are supported. The columns of ``sensitivities["all"]``
            follow ``Solution.sensitivity_names``, which is sorted. **Limitations**:
            sensitivities are not calculated up to numerical tolerances so are not
            guaranteed to be within the tolerances set by the solver, please raise an
            issue if you require this functionality. Also, when using this feature with
            `pybamm.Experiment`, the sensitivities do not take into account the movement
            of step-transitions wrt input parameters, so do not use this feature if the
            timings of your experimental protocol change rapidly with respect to your
            input parameters.
            Likewise, they do not include the movement of a discontinuity whose time
            depends on an input.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to None.
            Only valid for solvers that support intra-solve interpolation (`IDAKLUSolver`).
        state_mapper : tuple of (function, jacobian_y, jacobian_p), optional
            The output of `BaseSolver.process` for a function that maps the final state of the old solution
            to the initial state of `model`. Only `function` is used. If it is a CasADi function, sensitivities
            are mapped by forward-mode differentiation of it; otherwise they are matched by variable name.
            If not provided, then `BaseModel.set_initial_conditions_from` maps the state, and sensitivities are
            matched by variable name.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)
        :class:`pybamm.SolverError`
            If `dt`, `t_eval` or the period of a modulo of time is invalid, the old
            solution has no state sensitivities for an input in `calculate_sensitivities`,
            or `save` is True and the old solution has sensitivities for other inputs
        :class:`RuntimeError`
            If the solver has already been set up for a different model

        """

        # Set up inputs
        inputs = inputs or {}

        if old_solution is None:
            old_solution = pybamm.EmptySolution()

        if not (
            isinstance(old_solution, pybamm.EmptySolution)
            or old_solution.termination == "final time"
            or "[experiment]" in old_solution.termination
        ):
            # Return same solution as an event has already been triggered
            # With hack to allow stepping past experiment current / voltage cut-off
            return old_solution

        # Make sure model isn't empty
        self._check_empty_model(model)

        # Make sure dt is greater than zero
        if dt <= 0:
            raise pybamm.SolverError("Step time must be >0")

        # Raise deprecation warning for npts and convert it to t_eval
        if npts is not None:
            warnings.warn(
                "The 'npts' parameter is deprecated, use 't_eval' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            t_eval = np.linspace(0, dt, npts)
        elif t_eval is None:
            t_eval = np.array([0, dt])
        elif t_eval[0] != 0:
            raise pybamm.SolverError(
                f"The first `t_eval` value ({t_eval[0]}) must be 0."
                "Please correct your `t_eval` array."
            )
        elif t_eval[-1] != dt:
            raise pybamm.SolverError(
                f"The final `t_eval` value ({t_eval[-1]}) must be equal "
                f"to the step time `dt` ({dt}). Please correct your `t_eval` array."
            )
        else:
            pass

        t_interp = self.process_t_interp(t_interp, t_eval=t_eval)

        t_start = old_solution.t[-1]
        t_eval = t_start + t_eval
        t_interp = t_start + t_interp
        t_end = t_start + dt

        if t_start == 0:
            t_start_shifted = t_start
        else:
            # find the next largest floating point value for t_start
            # to avoid repeated times in the solution
            # from having the same time at the end of the previous step and
            # the start of the next step
            t_start_shifted = np.nextafter(t_start, np.inf)
            t_eval[0] = t_start_shifted
            if t_interp.size > 0 and t_interp[0] == t_start:
                t_interp[0] = t_start_shifted

        # Set timer
        timer = pybamm.Timer()

        # Set up inputs
        model_inputs = self._set_up_model_inputs(model, inputs)

        # Before the sensitivity setting below changes the model
        self._check_single_model(model)

        # process calculate_sensitivities argument
        previous_sensitivities = getattr(model, "calculate_sensitivities", [])
        sensitivity_names, sensitivities_have_changed = (
            BaseSolver._solve_process_calculate_sensitivities_arg(
                model_inputs, model, calculate_sensitivities, supplied_inputs=inputs
            )
        )
        try:
            if sensitivity_names:
                self._check_restart_sensitivities(old_solution, sensitivity_names)
            if save:
                self._check_saved_sensitivities(old_solution, sensitivity_names)
        except pybamm.SolverError:
            # Leave the model matching the solver's current set-up
            model.calculate_sensitivities = previous_sensitivities
            raise
        if sensitivities_have_changed:
            # Rebuild the solver for the new sensitivities, as solve() does
            self._model_set_up.pop(model, None)

        needs_set_up = model not in self._model_set_up
        if needs_set_up:
            self.set_up(model, model_inputs)
            self._model_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )

        if (
            isinstance(old_solution, pybamm.EmptySolution)
            and old_solution.termination is None
        ):
            pybamm.logger.verbose(f"Start stepping {model.name} with {self.name}")

        using_sensitivities = len(model.calculate_sensitivities) > 0

        if isinstance(old_solution, pybamm.EmptySolution):
            if not needs_set_up:
                # reset y0 to original initial conditions
                self.set_up(model, model_inputs, ics_only=True)
        elif old_solution.all_models[-1] == model:
            last_state = old_solution.last_state
            model.y0_list = [last_state.all_ys[0]]
            if using_sensitivities:
                sensitivities = self._restart_state_sensitivities(
                    old_solution, model.calculate_sensitivities
                )
                model.y0S_list = [tuple(sensitivities.T)]

        else:
            mapper_func = None
            if state_mapper is not None:
                last_state = old_solution.last_state
                y_from = last_state.all_ys[0]
                mapper_func = state_mapper[0]
                # Mapper CasADi graph is compiled using previous_model.input_parameters
                # (see Simulation._build_experiment_state_mappers); stack p in that
                # layout, not the next model's model_inputs.
                previous_model = old_solution.all_models[-1]
                # Legacy experiment solves often omit temperature in `inputs`
                # (include_temperature=False). Padding-rest models still list ambient
                # as an input parameter; match compile-time defaults from
                # Simulation._build_experiment_state_mappers (inputs.get(name, 0)).
                mapper_p_inputs = dict(inputs)
                for name in sorted(ip.name for ip in previous_model.input_parameters):
                    if name not in mapper_p_inputs or mapper_p_inputs[name] is None:
                        mapper_p_inputs[name] = inputs.get(name, 0)
                mapper_inputs = BaseSolver._set_up_model_inputs(
                    previous_model, mapper_p_inputs
                )
                p_casadi_stacked = casadi.vertcat(*[p for p in mapper_inputs.values()])

                model.y0_list = [
                    mapper_func(
                        t_start_shifted,
                        y_from,
                        p_casadi_stacked,
                    )
                ]
            else:
                _, concatenated_initial_conditions = model.set_initial_conditions_from(
                    old_solution,
                    inputs=model_inputs,
                    return_type="ics",
                )
                model.y0_list = [
                    concatenated_initial_conditions.evaluate(0, inputs=model_inputs)
                ]

            if using_sensitivities and isinstance(mapper_func, casadi.Function):
                # Forward-mode AD of the mapper gives J_y @ dy/dp + J_p
                names = model.calculate_sensitivities
                sizes = [np.size(value) for value in mapper_inputs.values()]
                starts = dict(zip(mapper_inputs, np.cumsum([0, *sizes]), strict=False))
                input_seeds = np.zeros((sum(sizes), len(names)))
                for column, name in enumerate(names):
                    input_seeds[starts[name], column] = 1
                sensitivities = mapper_func.forward(len(names))(
                    t_start_shifted,
                    y_from,
                    p_casadi_stacked,
                    model.y0_list[0],
                    np.zeros((1, len(names))),
                    self._restart_state_sensitivities(old_solution, names),
                    input_seeds,
                )
                model.y0S_list = [tuple(sensitivities.full().T)]
            elif using_sensitivities:
                model.y0S_list = [
                    self._set_sens_initial_conditions_from(
                        old_solution, model, model_inputs
                    )
                ]

        # Process discontinuities, which can move the start to just after one
        start_indices, end_indices, t_eval = self._get_discontinuity_start_end_indices(
            model, model.y0_list[0], model_inputs, t_eval
        )
        if t_eval[0] != t_start_shifted:
            t_interp = np.where(t_interp == t_start_shifted, t_eval[0], t_interp)

        set_up_time = timer.time()

        # (Re-)calculate consistent initialization
        self._set_consistent_initialization(model, t_eval[0], [model_inputs])

        # Check consistent initialization doesn't violate events
        self._check_event_violation_on_initialisation(
            t_eval, model, model.y0, model_inputs
        )

        # Step
        pybamm.logger.verbose(f"Stepping for {t_eval[0]:.0f} < t < {t_end:.0f}")
        timer.reset()

        solution = self._integrate_segments(
            model, t_eval, start_indices, end_indices, [model_inputs], t_interp, timer
        )[0]
        solution.solve_time = timer.time()

        # Check if extrapolation occurred
        self.check_extrapolation(solution, model.events)
        # Identify the event that caused termination and update the solution to
        # include the event time and state
        solution, termination = self.get_termination_reason(solution, model.events)

        # Assign setup time
        solution.set_up_time = set_up_time

        # Report times
        pybamm.logger.verbose(f"Finish stepping {model.name} ({termination})")
        pybamm.logger.verbose(
            f"Set-up time: {solution.set_up_time}, Step time: {solution.solve_time} (of which integration time: {solution.integration_time}), "
            f"Total time: {solution.total_time}"
        )

        # Return solution
        if save is False:
            return solution
        else:
            return old_solution + solution

    @staticmethod
    def get_termination_reason(solution, events):
        """
        Identify the cause for termination. In particular, if the solver terminated
        due to an event, (try to) pinpoint which event was responsible. If an event
        occurs the event time and state are added to the solution object.
        Note that the current approach (evaluating all the events and then finding which
        one is smallest at the final timestep) is pretty crude, but is the easiest one
        that works for all the different solvers.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution object
        events : dict
            Dictionary of events
        """
        termination_events = [
            x for x in events if x.event_type == pybamm.EventType.TERMINATION
        ]

        match solution.termination:
            case "final time":
                return (
                    solution,
                    "the solver successfully reached the end of the integration interval",
                )
            case "failure":
                return (
                    solution,
                    "the solver failed to simulate",
                )

        # solution.termination == "event":
        pybamm.logger.debug("Start post-processing events")
        if solution.closest_event_idx is not None:
            solution.termination = (
                f"event: {termination_events[solution.closest_event_idx].name}"
            )
        else:
            # Get final event value
            final_event_values = {}

            for event in termination_events:
                final_event_values[event.name] = event.expression.evaluate(
                    solution.t_event,
                    solution.y_event,
                    inputs=solution.all_inputs[-1],
                )
            termination_event = min(final_event_values, key=final_event_values.get)

            # Check that it's actually an event
            if final_event_values[termination_event] > 0.1:  # pragma: no cover
                # Hard to test this
                raise pybamm.SolverError(
                    "Could not determine which event was triggered "
                    "(possibly due to NaNs)"
                )
            # Add the event to the solution object
            solution.termination = f"event: {termination_event}"
        # Update t, y and inputs to include event time and state
        # Note: if the final entry of t is equal to the event time we skip
        # this (having duplicate entries causes an error later in ProcessedVariable)
        if solution.t_event != solution.all_ts[-1][-1]:
            event_sol = pybamm.Solution(
                solution.t_event,
                solution.y_event,
                solution.all_models[-1],
                solution.all_inputs[-1],
                solution.t_event,
                solution.y_event,
                solution.termination,
                variables_returned=solution.variables_returned,
                options=solution.user_options,
            )
            event_sol.solve_time = 0
            event_sol.integration_time = 0
            solution = solution + event_sol

        pybamm.logger.debug("Finish post-processing events")
        return solution, solution.termination

    def check_extrapolation(self, solution, events):
        """
        Check if extrapolation occurred for any of the interpolants. Note that with the
        current approach (evaluating all the events at the solution times) some
        extrapolations might not be found if they only occurred for a small period of
        time.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution object
        events : dict
            Dictionary of events
        """
        if self.on_extrapolation == "ignore":
            return

        extrap_events = []

        # Add the event dictionary to the solution object
        solution.extrap_events = extrap_events

        # first pass: check if any events are extrapolation events
        if all(
            event.event_type != pybamm.EventType.INTERPOLANT_EXTRAPOLATION
            for event in events
        ):
            # no extrapolation events to check
            return

        # second pass: check if the extrapolation events are within the tolerance
        last_state = solution.last_state
        if solution.t_event:
            t = solution.t_event[0]
            y = solution.y_event[:, 0]
        else:
            t = last_state.all_ts[0][0]
            y = last_state.all_ys[0][:, 0]
        inputs = last_state.all_inputs[0]

        if isinstance(y, casadi.DM):
            y = y.full()

        for event in events:
            if (
                event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION
                and event.expression.evaluate(t, y, inputs=inputs) < self.extrap_tol
            ):
                extrap_events.append(event.name)

        if len(extrap_events) == 0:
            # no extrapolation events are within the tolerance
            return

        if self.on_extrapolation == "error":
            raise pybamm.SolverError(
                "Solver failed because the following "
                f"interpolation bounds were exceeded: {extrap_events}. "
                "You may need to provide additional interpolation points "
                "outside these bounds."
            )
        elif self.on_extrapolation == "warn":
            name = solution.all_models[-1].name
            warnings.warn(
                f"While solving {name} extrapolation occurred for {extrap_events}",
                pybamm.SolverWarning,
                stacklevel=2,
            )

    def _check_empty_model(self, model):
        # Make sure model isn't empty
        if (
            (len(model.rhs) == 0 and len(model.algebraic) == 0)
            and model.concatenated_rhs is None
            and model.concatenated_algebraic is None
            and not isinstance(self, pybamm.DummySolver)
        ):
            raise pybamm.ModelError(
                "Cannot simulate an empty model, use `pybamm.DummySolver` instead"
            )

    def get_platform_context(self, system_type: str):
        # Set context for parallel processing depending on the platform
        if system_type.lower() in ["linux"]:
            if pybamm.has_jax():
                # JAX is incompatible with fork (see https://github.com/jax-ml/jax/issues/1805)
                # Use spawn if JAX is available to avoid corruption of JAX's global state
                return "spawn"
            else:
                return "fork"
        return "spawn"

    @staticmethod
    def _set_up_model_inputs(model: pybamm.BaseModel, inputs: dict):
        """Set up input parameters"""
        inputs = ParameterValues.check_parameter_values(inputs)

        # First, check all the strictly required input parameters that must be included
        # for the model to be solvable.
        inputs_in_model = {}
        missing_required_inputs = False

        required_input_parameters = {ip.name for ip in model.required_input_parameters}

        for name in required_input_parameters:
            value = inputs.get(name)
            missing_required_inputs |= value is None
            if name and value is not None:
                inputs_in_model[name] = value

        input_parameters = {ip.name for ip in model.input_parameters}

        if missing_required_inputs:
            ip = sorted(input_parameters)
            missing = sorted(set(ip) - set(inputs.keys()))

            ip_str = ", ".join(f"'{n}'" for n in ip)
            missing_str = ", ".join(f"'{n}'" for n in missing)
            raise pybamm.SolverError(
                f"No value provided for input: [{missing_str}].\n"
                f"Required inputs: [{ip_str}].\n"
            )

        # Next, check the full set of specified input parameters that are required for
        # the model to be observable.
        missing_inputs = []
        solution_is_observable = model.solution_observable
        for name in input_parameters:
            value = inputs.get(name)

            if value is None:
                # Since the solver has all the strictly required input parameters, but
                # lacks the total set of input parameters, the model is solvable, but
                # unobservable. In this case, we can safely set the input parameter to
                # a dummy value (np.nan)
                missing_inputs.append(name)
                value = DUMMY_INPUT_PARAMETER_VALUE
                model.disable_solution_observability(
                    ModelSolutionObservability.MISSING_INPUT_PARAMETERS
                )

            if name not in inputs_in_model:
                inputs_in_model[name] = value

        if solution_is_observable and missing_inputs:
            missing_inputs_str = ", ".join(f"'{n}'" for n in missing_inputs)
            pybamm.logger.warning(
                f"No value provided for input: [{missing_inputs_str}]. "
                "The `Solution` can no longer be observed with `solution.observe(symbol)`. "
                "To fix this, provide all input parameter values to `simulation.solve(..., inputs=inputs)` "
                "or `simulation.step(..., inputs=inputs)`.",
            )

        ordered_inputs_names = sorted(inputs_in_model.keys())
        ordered_inputs = {name: inputs_in_model[name] for name in ordered_inputs_names}

        return ordered_inputs


def flatten_inputs(inputs_dict: dict) -> npt.NDArray[np.float64]:
    """Flatten ``{name: value}`` into one 1D array.

    Values are concatenated in dict-key order, and a vector-valued input
    contributes its whole block.

    Parameters
    ----------
    inputs_dict : dict
        Input values for one input set.

    Returns
    -------
    :class:`numpy.ndarray`
        Flat ``float64`` vector, empty for an empty dict.
    """
    if not inputs_dict:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()
            for v in inputs_dict.values()
        ]
    )


def stack_inputs(inputs_dict: dict) -> casadi.DM:
    """Stack ``{name: value}`` into one column vector, in dict-key order.

    Parameters
    ----------
    inputs_dict : dict
        Input values for one input set.

    Returns
    -------
    :class:`casadi.DM`
        Column vector of the input values, with zero rows for an empty dict.
    """
    return casadi.vertcat(*inputs_dict.values())


def process(
    symbol, name, vars_for_processing, use_jacobian=None, return_jacp_stacked=None
):
    """
    Parameters
    ----------
    symbol: :class:`pybamm.Symbol`
        expression tree to convert
    name: str
        function evaluators created will have this base name
    vars_for_processing: dict
        dictionary of variables for processing
    use_jacobian: bool, optional
        whether to return Jacobian functions
    return_jacp_stacked: bool, optional
        returns Jacobian function wrt the stacked sensitivity parameters instead
        of jacp

    Returns
    -------
    func: :class:`pybamm.EvaluatorPython` or
            :class:`pybamm.EvaluatorJax` or
            :class:`casadi.Function`
        evaluator for the function $f(y, t, p)$ given by `symbol`

    jac: :class:`pybamm.EvaluatorPython` or
            :class:`pybamm.EvaluatorJaxJacobian` or
            :class:`casadi.Function`
        evaluator for the Jacobian $\\frac{\\partial f}{\\partial y}$
        of the function given by `symbol`

    jacp: :class:`pybamm.EvaluatorPython` or
            :class:`pybamm.EvaluatorJaxSensitivities` or
            :class:`casadi.Function`
        evaluator for the parameter sensitivities
        $\frac{\\partial f}{\\partial p}$
        of the function given by `symbol`

    jac_action: :class:`pybamm.EvaluatorPython` or
            :class:`pybamm.EvaluatorJax` or
            :class:`casadi.Function`
        evaluator for product of the Jacobian with a vector $v$,
        i.e. $\\frac{\\partial f}{\\partial y} * v$
    """

    def report(string):
        # don't log event conversion
        if "event" not in string:
            pybamm.logger.verbose(string)

    model = vars_for_processing["model"]

    if use_jacobian is None:
        use_jacobian = model.use_jacobian

    if model.convert_to_format == "jax":
        report(f"Converting {name} to jax")
        func = pybamm.EvaluatorJax(symbol)
        jacp = None
        if model.calculate_sensitivities:
            report(
                f"Calculating sensitivities for {name} with respect "
                f"to parameters {model.calculate_sensitivities} using jax"
            )
            jacp = func.get_sensitivities()
        if use_jacobian:
            report(f"Calculating jacobian for {name} using jax")
            jac = func.get_jacobian()
            jac_action = func.get_jacobian_action()
        else:
            jac = None
            jac_action = None

    elif model.convert_to_format != "casadi":
        y = vars_for_processing["y"]
        jacobian = vars_for_processing["jacobian"]

        if model.calculate_sensitivities:
            raise pybamm.SolverError(  # pragma: no cover
                "Sensitivies are no longer supported for the python "
                "evaluator. Please use `convert_to_format = 'casadi'`, or `jax` "
                "to calculate sensitivities."
            )

        else:
            jacp = None

        if use_jacobian:
            report(f"Calculating jacobian for {name}")
            jac = jacobian.jac(symbol, y)
            report(f"Converting jacobian for {name} to python")
            jac = pybamm.EvaluatorPython(jac)
            # cannot do jacobian action efficiently for now
            jac_action = None
        else:
            jac = None
            jac_action = None

        report(f"Converting {name} to python")
        func = pybamm.EvaluatorPython(symbol)

    else:
        t_casadi = vars_for_processing["t_casadi"]
        y_casadi = vars_for_processing["y_casadi"]
        p_casadi = vars_for_processing["p_casadi"]
        p_casadi_stacked = vars_for_processing["p_casadi_stacked"]

        # Process with CasADi
        report(f"Converting {name} to CasADi")
        casadi_expression = symbol.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
        # Add sensitivity vectors to the rhs and algebraic equations
        jacp = None

        if model.calculate_sensitivities:
            report(
                f"Calculating sensitivities for {name} with respect "
                f"to parameters {model.calculate_sensitivities} using "
                "CasADi"
            )
            # One column per sensitivity input, in the solver's order
            if return_jacp_stacked:
                sensitivity_inputs_stacked = casadi.vertcat(
                    *[p_casadi[pname] for pname in model.calculate_sensitivities]
                )
                jacp = casadi.Function(
                    f"d{name}_dp",
                    [t_casadi, y_casadi, p_casadi_stacked],
                    [casadi.jacobian(casadi_expression, sensitivity_inputs_stacked)],
                )
            else:
                # WARNING, jacp for convert_to_format=casadi does not return a dict
                # instead it returns multiple return values, one for each param
                # TODO: would it be faster to do the jacobian wrt pS_casadi_stacked?
                jacp = casadi.Function(
                    name + "_jacp",
                    [t_casadi, y_casadi, p_casadi_stacked],
                    [
                        casadi.densify(
                            casadi.jacobian(casadi_expression, p_casadi[pname])
                        )
                        for pname in model.calculate_sensitivities
                    ],
                )

        if use_jacobian:
            report(f"Calculating jacobian for {name} using CasADi")
            # The model owns how its jacobian is built (plain CasADi AD unless the model
            # has a switching control equation it must differentiate structurally).
            jac_casadi = model.build_casadi_jacobian(casadi_expression, y_casadi)
            jac = casadi.Function(
                name + "_jac",
                [t_casadi, y_casadi, p_casadi_stacked],
                [jac_casadi],
            )

            v = casadi.MX.sym(
                "v",
                model.len_rhs_and_alg,
            )
            jac_action_casadi = casadi.densify(
                casadi.jtimes(casadi_expression, y_casadi, v)
            )
            jac_action = casadi.Function(
                name + "_jac_action",
                [t_casadi, y_casadi, p_casadi_stacked, v],
                [jac_action_casadi],
            )
        else:
            jac = None
            jac_action = None

        func = casadi.Function(
            name, [t_casadi, y_casadi, p_casadi_stacked], [casadi_expression]
        )

    return func, jac, jacp, jac_action
