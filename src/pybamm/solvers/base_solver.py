import copy
import itertools
import multiprocessing as mp
import numbers
import platform
import sys
import warnings

import casadi
import numpy as np

import pybamm
from pybamm import ParameterValues
from pybamm.expression_tree.binary_operators import _Heaviside
from pybamm.expression_tree.input_parameter import DUMMY_INPUT_PARAMETER_VALUE
from pybamm.models.base_model import ModelSolutionObservability


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
        If a solver class, must be an algebraic solver class.
        If "casadi",
        the solver uses casadi's Newton rootfinding algorithm to find initial
        conditions. Otherwise, the solver uses 'scipy.optimize.root' with method
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
        Default is "raise".
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
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.root_tol = root_tol
        self.root_method = root_method
        self.extrap_tol = extrap_tol or -1e-10
        self.output_variables = [] if output_variables is None else output_variables
        self._on_extrapolation = on_extrapolation or "warn"
        self._on_failure = on_failure or "raise"
        self._model_set_up = {}

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self._ode_solver = False
        self._algebraic_solver = False
        self._supports_interp = False
        self._supports_t_eval_discontinuities = False
        self.computed_var_fcns = {}
        self._mp_context = self.get_platform_context(platform.system())

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
            raise ValueError("on_extrapolation must be 'warn', 'raise', or 'ignore'")
        self._on_extrapolation = value

    @property
    def on_failure(self):
        return self._on_failure

    @on_failure.setter
    def on_failure(self, value):
        if value not in ["warn", "error", "ignore"]:
            raise ValueError("on_failure must be 'warn', 'raise', or 'ignore'")
        self._on_failure = value

    @property
    def root_method(self):
        return self._root_method

    @root_method.setter
    def root_method(self, method):
        if method == "casadi":
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
            The times at which to stop the integration due to a discontinuity in time.
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
        ) = self._set_up_events(model, t_eval, inputs, vars_for_processing)

        # Add the solver attributes
        model.rhs_eval = rhs
        model.algebraic_eval = algebraic
        model.rhs_algebraic_eval = rhs_algebraic

        model.terminate_events_eval = terminate_events
        model.interpolant_extrapolation_events_eval = interpolant_extrapolation_events
        model.discontinuity_events_eval = discontinuity_events
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
            self, pybamm.CasadiSolver | pybamm.CasadiAlgebraicSolver
        ):
            # can use DAE solver to solve model with algebraic equations only
            if len(model.rhs) > 0:
                t_casadi = vars_for_processing["t_casadi"]
                y_casadi = vars_for_processing["y_casadi"]
                p_casadi_stacked = vars_for_processing["p_casadi_stacked"]
                mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
                explicit_rhs = mass_matrix_inv @ rhs(
                    t_casadi, y_casadi, p_casadi_stacked
                )
                model.casadi_rhs = casadi.Function(
                    "rhs", [t_casadi, y_casadi, p_casadi_stacked], [explicit_rhs]
                )
            model.casadi_switch_events = casadi_switch_events
            model.casadi_algebraic = algebraic
            model.casadi_sensitivities = jacp_rhs_algebraic
            model.casadi_sensitivities_rhs = jacp_rhs
            model.casadi_sensitivities_algebraic = jacp_algebraic

        if getattr(self.root_method, "algebraic_solver", False):
            self.root_method.set_up_root_solver(model, inputs[0], t_eval)

        # if output_variables specified then convert functions to casadi
        # expressions for evaluation within the respective solver
        self.computed_var_fcns = {}
        self.computed_dvar_dy_fcns = {}
        self.computed_dvar_dp_fcns = {}
        self._time_integral_vars = {}
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
                BaseSolver._wrangle_name(key),
                vars_for_processing,
                use_jacobian=True,
                return_jacp_stacked=True,
            )

        pybamm.logger.info("Finish solver set-up")

    def _set_initial_conditions(self, model, time, inputs: list[dict]):
        # model should have been discretised or an error raised in Self._check_and_prepare_model_inplace
        len_tot = model.len_rhs_and_alg
        y_zero = np.zeros((len_tot, 1))

        casadi_format = model.convert_to_format == "casadi"
        model.y0_list = []
        model.y0S_list = [] if model.jacp_initial_conditions_eval is not None else None
        for ipts in inputs:
            if casadi_format:
                # stack inputs
                inputs_y0_ics = casadi.vertcat(*[x for x in ipts.values()])
            else:
                inputs_y0_ics = ipts

            model.y0_list.append(
                model.initial_conditions_eval(time, y_zero, inputs_y0_ics)
            )

            if model.jacp_initial_conditions_eval is not None:
                if casadi_format:
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
        Wrangle a function name to replace special characters
        """
        replacements = [
            (" ", "_"),
            ("[", ""),
            ("]", ""),
            (".", "_"),
            ("-", "_"),
            ("(", ""),
            (")", ""),
            ("%", "prc"),
            (",", ""),
            (".", ""),
        ]
        name = "v_" + name.casefold()
        for string, replacement in replacements:
            name = name.replace(string, replacement)
        return name

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
        has_mass_matrix = model.mass_matrix is not None
        has_mass_matrix_inv = model.mass_matrix_inv is not None

        if not has_mass_matrix:
            return

        model.mass_matrix = pybamm.Matrix(
            model.mass_matrix.entries[: model.len_rhs_and_alg, : model.len_rhs_and_alg]
        )
        if has_mass_matrix_inv:
            model.mass_matrix_inv = pybamm.Matrix(
                model.mass_matrix_inv.entries[: model.len_rhs, : model.len_rhs]
            )

    def _set_up_events(self, model, t_eval, inputs: list[dict], vars_for_processing):
        # Check for heaviside and modulo functions in rhs and algebraic and add
        # discontinuity events if these exist.
        # Note: only checks for the case of t < X, t <= X, X < t, or X <= t,
        # but also accounts for the fact that t might be dimensional
        tf = np.max(t_eval)

        def supports_t_eval_discontinuities(expr):
            # Only IDAKLUSolver supports discontinuities represented by t_eval
            return self.supports_t_eval_discontinuities and expr.is_constant()

        # Find all the constant time-based discontinuities
        t_discon_symbols = []

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
                    t_discon_symbols.append((symbol, expr, None))
                else:
                    model.events.append(
                        pybamm.Event(
                            str(symbol),
                            expr,
                            pybamm.EventType.DISCONTINUITY,
                        )
                    )

            elif isinstance(symbol, pybamm.Modulo) and symbol.left == pybamm.t:
                expr = symbol.right
                num_events = 200 if (t_eval is None) else (tf // expr.value)

                if supports_t_eval_discontinuities(expr):
                    t_discon_symbols.append((symbol, expr, num_events))
                else:
                    for i in np.arange(num_events):
                        model.events.append(
                            pybamm.Event(
                                str(symbol),
                                expr * pybamm.Scalar(i + 1),
                                pybamm.EventType.DISCONTINUITY,
                            )
                        )
            else:
                continue

        casadi_switch_events = []
        terminate_events = []
        interpolant_extrapolation_events = []
        discontinuity_events = []
        for n, event in enumerate(model.events):
            if event.event_type == pybamm.EventType.DISCONTINUITY:
                # discontinuity events are evaluated before the solver is called,
                # so don't need to process them
                discontinuity_events.append(event)
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

    def _solve_process_calculate_sensitivities_arg(
        inputs: dict, model: pybamm.BaseModel, calculate_sensitivities: list[str] | bool
    ):
        # get a list-only version of calculate_sensitivities
        if isinstance(calculate_sensitivities, bool):
            if calculate_sensitivities:
                calculate_sensitivities_list = [p for p in inputs.keys()]
            else:
                calculate_sensitivities_list = []
        else:
            calculate_sensitivities_list = calculate_sensitivities

        calculate_sensitivities_list.sort()
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        # Check that calculate_sensitivites have not been updated
        sensitivities_have_changed = (
            calculate_sensitivities_list != model.calculate_sensitivities
        )

        # save sensitivity parameters so we can identify them later on
        # (FYI: this is used in the Solution class)
        model.calculate_sensitivities = calculate_sensitivities_list

        return calculate_sensitivities_list, sensitivities_have_changed

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

        ninputs = len(inputs_list)
        if ninputs == 1:
            new_solution = self._integrate_single(
                model,
                t_eval,
                inputs_list[0],
                model.y0_list[0],
            )
            new_solutions = [new_solution]
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
            Whether the solver calculates sensitivities of all input parameters. Defaults to False.
            If only a subset of sensitivities are required, can also pass a
            list of input parameter names.  **Limitations**: sensitivities are not calculated up to numerical tolerances
            so are not guarenteed to be within the tolerances set by the solver, please raise an issue if you
            require this functionality. Also, when using this feature with `pybamm.Experiment`, the sensitivities
            do not take into account the movement of step-transitions wrt input parameters, so do not use this feature
            if the timings of your experimental protocol change rapidly with respect to your input parameters.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to None.
            Only valid for solvers that support intra-solve interpolation (`IDAKLUSolver`).
        Returns
        -------
        :class:`pybamm.Solution` or list of :class:`pybamm.Solution` objects.
             If type of `inputs` is `list`, return a list of corresponding
             :class:`pybamm.Solution` objects.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}` and
            `model.variables = {}`)
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

        t_interp = self.process_t_interp(t_interp)

        # Set up inputs
        if isinstance(inputs, dict):
            inputs_list = [inputs]
        else:
            inputs_list = inputs or [{}]

        model_inputs_list: list[dict] = [
            self._set_up_model_inputs(model, inputs) for inputs in inputs_list
        ]

        _, sensitivities_have_changed = (
            BaseSolver._solve_process_calculate_sensitivities_arg(
                model_inputs_list[0], model, calculate_sensitivities
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
            if len(self._model_set_up) > 0:
                existing_model = next(iter(self._model_set_up))
                raise RuntimeError(
                    "This solver has already been initialised for model "
                    f'"{existing_model.name}". Please create a separate '
                    "solver for this model"
                )
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

        # Solve for the consistent initialization
        self._set_consistent_initialization(model, t_eval[0], model_inputs_list)

        set_up_time = timer.time()
        timer.reset()

        # Check initial conditions don't violate events
        for y0, inpts in zip(model.y0_list, model_inputs_list, strict=True):
            self._check_events_with_initialization(t_eval, model, y0, inpts)

        # Process discontinuities
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

        start_indices, end_indices, t_eval = first_row

        # Integrate separately over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a DAE)
        old_y0_list = model.y0_list
        solutions = None
        for start_index, end_index in zip(start_indices, end_indices, strict=False):
            pybamm.logger.verbose(
                f"Calling solver for {t_eval[start_index]} < t < {t_eval[end_index - 1]}"
            )
            new_solutions = self._integrate(
                model,
                t_eval[start_index:end_index],
                model_inputs_list,
                t_interp,
                nproc=nproc,
            )
            # Setting the solve time for each segment.
            # pybamm.Solution.__add__ assumes attribute solve_time.
            solve_time = timer.time()
            for sol in new_solutions:
                sol.solve_time = solve_time
            if start_index == start_indices[0]:
                solutions = [sol for sol in new_solutions]
            else:
                for i, new_solution in enumerate(new_solutions):
                    solutions[i] = solutions[i] + new_solution

            if solutions[0].termination != "final time":
                break

            if end_index != len(t_eval):
                # setup for next integration subsection
                for i, soln in enumerate(new_solutions):
                    last_state = soln.y[:, -1]
                    # update y0 (for DAE solvers, this updates the initial guess for the
                    # rootfinder)
                    model.y0_list[i] = last_state
                if len(model.algebraic) > 0:
                    model.y0_list = self.calculate_consistent_state(
                        model, t_eval[end_index], model_inputs_list
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

        # Restore old y0
        model.y0_list = old_y0_list

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
        # solvers, where we may only expect one time in the solution)
        if (
            self._algebraic_solver is False
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

    def _get_discontinuity_start_end_indices(self, model, y0, inputs, t_eval):
        if self.supports_t_eval_discontinuities and model.t_discon_constant_symbols:
            pybamm.logger.verbose("Discontinuity events found for constant symbols")
            _t_discon_constant = []
            for symbol, expr, num_events in model.t_discon_constant_symbols:
                _t_discon_constant.extend(
                    symbol._t_discon(expr, y0, inputs, num_events)
                )

            t_discon_constant = self.filter_discontinuities(_t_discon_constant, t_eval)
            t_eval = np.union1d(t_eval, t_discon_constant)

        if not model.discontinuity_events_eval:
            pybamm.logger.verbose("No additional discontinuity events found")
            return [0], [len(t_eval)], t_eval

        # Calculate all possible discontinuities
        _t_discon_full = [
            event.expression.evaluate(inputs=inputs)
            for event in model.discontinuity_events_eval
        ]
        t_discon = self.filter_discontinuities(_t_discon_full, t_eval)

        pybamm.logger.verbose(f"Discontinuity events found at t = {t_discon}")

        # insert time points around discontinuities in t_eval
        # keep track of subsections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        eps = sys.float_info.epsilon
        for dtime in t_discon:
            dindex = np.searchsorted(t_eval, dtime, side="left")
            end_indices.append(dindex + 1)
            start_indices.append(dindex + 1)
            if dtime * (1 - eps) < t_eval[dindex] < dtime * (1 + eps):
                t_eval[dindex] *= 1 + eps
                t_eval = np.insert(t_eval, dindex, dtime * (1 - eps))
            else:
                t_eval = np.insert(
                    t_eval, dindex, [dtime * (1 - eps), dtime * (1 + eps)]
                )
        end_indices.append(len(t_eval))

        return start_indices, end_indices, t_eval

    @staticmethod
    def _check_events_with_initialization(t_eval, model, y0, inputs_dict):
        num_terminate_events = len(model.terminate_events_eval)
        if num_terminate_events == 0:
            return

        if model.convert_to_format == "casadi":
            inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        events_eval = np.empty(num_terminate_events)
        for idx, event in enumerate(model.terminate_events_eval):
            if model.convert_to_format == "casadi":
                event_eval = event(t_eval[0], y0, inputs)
            elif model.convert_to_format in ["python", "jax"]:
                event_eval = event(t=t_eval[0], y=y0, inputs=inputs_dict)
                if not isinstance(event_eval, float):
                    event_eval = event_eval.item()
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

    def _set_sens_initial_conditions_from(
        self, solution: pybamm.Solution, model: pybamm.BaseModel
    ) -> tuple:
        """
        A restricted version of BaseModel.set_initial_conditions_from that only extracts the
        sensitivities from a solution object, and only for a model that has been discretised.
        This is used when setting the initial conditions for a sensitivity model.

        Parameters
        ----------
        solution : :class:`pybamm.Solution`
            The solution to use to initialize the model

        model: :class:`pybamm.BaseModel`
            The model whose sensitivities to set

        Returns
        -------

        initial_conditions : tuple of ndarray
            The initial conditions for the sensitivities, each element of the tuple
            corresponds to an input parameter
        """

        ninputs = len(model.calculate_sensitivities)
        initial_conditions = tuple([] for _ in range(ninputs))
        solution = solution.last_state
        for var in model.initial_conditions:
            final_state = solution[var.name]
            final_state = final_state.sensitivities
            final_state_eval = tuple(
                final_state[key] for key in model.calculate_sensitivities
            )

            scale, reference = var.scale.value, var.reference.value
            for i in range(ninputs):
                scaled_final_state_eval = (final_state_eval[i] - reference) / scale
                initial_conditions[i].append(scaled_final_state_eval)

        # Also update the concatenated initial conditions if the model is already
        # discretised
        # Unpack slices for sorting
        y_slices = {var: slce for var, slce in model.y_slices.items()}
        slices = [y_slices[symbol][0] for symbol in model.initial_conditions.keys()]

        # sort equations according to slices
        concatenated_initial_conditions = [
            casadi.vertcat(*[eq for _, eq in sorted(zip(slices, init, strict=True))])
            for init in initial_conditions
        ]
        return concatenated_initial_conditions

    def process_t_interp(self, t_interp):
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
            By default, the solution is returned at t0 and t0 + dt.
        npts : deprecated
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool, optional
            Save solution with all previous timesteps. Defaults to True.
        calculate_sensitivities : list of str or bool, optional
            Whether the solver calculates sensitivities of all input parameters. Defaults to False.
            If only a subset of sensitivities are required, can also pass a
            list of input parameter names. **Limitations**: sensitivities are not calculated up to numerical tolerances
            so are not guaranteed to be within the tolerances set by the solver, please raise an issue if you
            require this functionality. Also, when using this feature with `pybamm.Experiment`, the sensitivities
            do not take into account the movement of step-transitions wrt input parameters, so do not use this feature
            if the timings of your experimental protocol change rapidly with respect to your input parameters.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to None.
            Only valid for solvers that support intra-solve interpolation (`IDAKLUSolver`).
        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

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

        t_interp = self.process_t_interp(t_interp)

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

        # process calculate_sensitivities argument
        _, sensitivities_have_changed = (
            BaseSolver._solve_process_calculate_sensitivities_arg(
                model_inputs, model, calculate_sensitivities
            )
        )

        first_step_this_model = model not in self._model_set_up
        if first_step_this_model or sensitivities_have_changed:
            if len(self._model_set_up) > 0:
                existing_model = next(iter(self._model_set_up))
                raise RuntimeError(
                    "This solver has already been initialised for model "
                    f'"{existing_model.name}". Please create a separate '
                    "solver for this model"
                )
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
            if not first_step_this_model:
                # reset y0 to original initial conditions
                self.set_up(model, model_inputs, ics_only=True)
        elif old_solution.all_models[-1] == model:
            last_state = old_solution.last_state
            model.y0_list = [last_state.all_ys[0]]
            if using_sensitivities:
                full_sens = last_state._all_sensitivities["all"][0]
                model.y0S_list = [
                    tuple(full_sens[:, i] for i in range(full_sens.shape[1]))
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

            if using_sensitivities:
                model.y0S_list = [
                    self._set_sens_initial_conditions_from(old_solution, model)
                ]

        set_up_time = timer.time()

        # (Re-)calculate consistent initialization
        self._set_consistent_initialization(model, t_start_shifted, [model_inputs])

        # Check consistent initialization doesn't violate events
        self._check_events_with_initialization(t_eval, model, model.y0, model_inputs)

        # Step
        pybamm.logger.verbose(f"Stepping for {t_start_shifted:.0f} < t < {t_end:.0f}")
        timer.reset()

        solution = self._integrate(model, t_eval, [model_inputs], t_interp)[0]
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
        returns Jacobian function wrt stacked parameters instead of jacp

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
            # Compute derivate wrt p-stacked (can be passed to solver to
            # compute sensitivities online)
            if return_jacp_stacked:
                jacp = casadi.Function(
                    f"d{name}_dp",
                    [t_casadi, y_casadi, p_casadi_stacked],
                    [casadi.jacobian(casadi_expression, p_casadi_stacked)],
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
            jac_casadi = casadi.jacobian(casadi_expression, y_casadi)
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
