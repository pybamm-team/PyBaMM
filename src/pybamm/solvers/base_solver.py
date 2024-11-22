import copy
import itertools
from scipy.sparse import block_diag
import multiprocessing as mp
import numbers
import sys
import warnings
import platform

import casadi
import numpy as np

import pybamm
from pybamm.expression_tree.binary_operators import _Heaviside
from pybamm import ParameterValues


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
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method=None,
        root_tol=1e-6,
        extrap_tol=None,
        output_variables=None,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.root_tol = root_tol
        self.root_method = root_method
        self.extrap_tol = extrap_tol or -1e-10
        self.output_variables = [] if output_variables is None else output_variables
        self._model_set_up = {}

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self._ode_solver = False
        self._algebraic_solver = False
        self._supports_interp = False
        self._on_extrapolation = "warn"
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
    def root_method(self):
        return self._root_method

    @property
    def supports_parallel_solve(self):
        return False

    @property
    def requires_explicit_sensitivities(self):
        return True

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

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False):
        """Unpack model, perform checks, and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        t_eval : numeric type, optional
            The times at which to stop the integration due to a discontinuity in time.
        """
        inputs = inputs or {}

        if ics_only:
            pybamm.logger.info("Start solver set-up, initial_conditions only")
        else:
            pybamm.logger.info("Start solver set-up")

        self._check_and_prepare_model_inplace(model, inputs, ics_only)

        # set default calculate sensitivities on model
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        # see if we need to form the explicit sensitivity equations
        calculate_sensitivities_explicit = (
            model.calculate_sensitivities and self.requires_explicit_sensitivities
        )

        self._set_up_model_sensitivities_inplace(
            model, inputs, calculate_sensitivities_explicit
        )

        vars_for_processing = self._get_vars_for_processing(
            model, inputs, calculate_sensitivities_explicit
        )

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
            discontinuity_events,
        ) = self._set_up_events(model, t_eval, inputs, vars_for_processing)

        # Add the solver attributes
        model.rhs_eval = rhs
        model.algebraic_eval = algebraic
        model.rhs_algebraic_eval = rhs_algebraic

        model.terminate_events_eval = terminate_events
        model.discontinuity_events_eval = discontinuity_events
        model.interpolant_extrapolation_events_eval = interpolant_extrapolation_events

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
            self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver)
        ):
            # can use DAE solver to solve model with algebraic equations only
            if len(model.rhs) > 0:
                t_casadi = vars_for_processing["t_casadi"]
                y_and_S = vars_for_processing["y_and_S"]
                p_casadi_stacked = vars_for_processing["p_casadi_stacked"]
                mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
                explicit_rhs = mass_matrix_inv @ rhs(
                    t_casadi, y_and_S, p_casadi_stacked
                )
                model.casadi_rhs = casadi.Function(
                    "rhs", [t_casadi, y_and_S, p_casadi_stacked], [explicit_rhs]
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
        for key in self.output_variables:
            # ExplicitTimeIntegral's are not computed as part of the solver and
            # do not need to be converted
            if isinstance(model.variables_and_events[key], pybamm.ExplicitTimeIntegral):
                continue
            # Generate Casadi function to calculate variable and derivates
            # to enable sensitivites to be computed within the solver
            (
                self.computed_var_fcns[key],
                self.computed_dvar_dy_fcns[key],
                self.computed_dvar_dp_fcns[key],
                _,
            ) = process(
                model.variables_and_events[key],
                BaseSolver._wrangle_name(key),
                vars_for_processing,
                use_jacobian=True,
                return_jacp_stacked=True,
            )

        pybamm.logger.info("Finish solver set-up")

    def _set_initial_conditions(self, model, time, inputs):
        len_tot = model.len_rhs_and_alg + model.len_rhs_sens + model.len_alg_sens
        y_zero = np.zeros((len_tot, 1))

        casadi_format = model.convert_to_format == "casadi"
        if casadi_format:
            # stack inputs
            inputs_y0_ics = casadi.vertcat(*[x for x in inputs.values()])
        else:
            inputs_y0_ics = inputs

        model.y0 = model.initial_conditions_eval(time, y_zero, inputs_y0_ics)

        if model.jacp_initial_conditions_eval is None:
            model.y0S = None
            return

        if casadi_format:
            inputs_jacp_ics = inputs_y0_ics
        else:
            # we are calculating the derivative wrt the inputs
            # so need to make sure we convert int -> float
            # This is to satisfy JAX jacfwd function which requires
            # float inputs
            inputs_jacp_ics = {
                key: float(value) if isinstance(value, int) else value
                for key, value in inputs.items()
            }

        model.y0S = model.jacp_initial_conditions_eval(time, y_zero, inputs_jacp_ics)

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

    def _check_and_prepare_model_inplace(self, model, inputs, ics_only):
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
            isinstance(self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver))
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
    def _get_vars_for_processing(model, inputs, calculate_sensitivities_explicit):
        vars_for_processing = {
            "model": model,
            "calculate_sensitivities_explicit": calculate_sensitivities_explicit,
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
            # sensitivity vectors
            if calculate_sensitivities_explicit:
                pS_casadi_stacked = casadi.vertcat(
                    *[p_casadi[name] for name in model.calculate_sensitivities]
                )
                S_x = casadi.MX.sym("S_x", model.len_rhs_sens)
                S_z = casadi.MX.sym("S_z", model.len_alg_sens)
                vars_for_processing.update(
                    {"S_x": S_x, "S_z": S_z, "pS_casadi_stacked": pS_casadi_stacked}
                )
                y_and_S = casadi.vertcat(y_diff, S_x, y_alg, S_z)
            else:
                y_and_S = y_casadi
            vars_for_processing.update({"y_and_S": y_and_S})

            return vars_for_processing

    @staticmethod
    def _set_up_model_sensitivities_inplace(
        model, inputs, calculate_sensitivities_explicit
    ):
        """
        Set up model attributes related to sensitivities.
        """
        # if we are calculating sensitivities explicitly then the number of
        # equations will change
        if calculate_sensitivities_explicit:
            num_parameters = 0
            for name in model.calculate_sensitivities:
                # if not a number, assume its a vector
                if isinstance(inputs[name], numbers.Number):
                    num_parameters += 1
                else:
                    num_parameters += len(inputs[name])
            model.len_rhs_sens = model.len_rhs * num_parameters
            model.len_alg_sens = model.len_alg * num_parameters
        else:
            model.len_rhs_sens = 0
            model.len_alg_sens = 0

        has_mass_matrix = model.mass_matrix is not None
        has_mass_matrix_inv = model.mass_matrix_inv is not None

        if not has_mass_matrix:
            return

        # if we will change the equations to include the explicit sensitivity
        # equations, then we also need to update the mass matrix and bounds.
        # First, we reset the mass matrix and bounds back to their original form
        # if they have been extended
        if model.bounds[0].shape[0] > model.len_rhs_and_alg:
            model.bounds = (
                model.bounds[0][: model.len_rhs_and_alg],
                model.bounds[1][: model.len_rhs_and_alg],
            )
        model.mass_matrix = pybamm.Matrix(
            model.mass_matrix.entries[: model.len_rhs_and_alg, : model.len_rhs_and_alg]
        )
        if has_mass_matrix_inv:
            model.mass_matrix_inv = pybamm.Matrix(
                model.mass_matrix_inv.entries[: model.len_rhs, : model.len_rhs]
            )

        # now we can extend them by the number of sensitivity parameters
        # if necessary
        if not calculate_sensitivities_explicit:
            return

        if model.bounds[0].shape[0] == model.len_rhs_and_alg:
            model.bounds = (
                np.repeat(model.bounds[0], num_parameters + 1),
                np.repeat(model.bounds[1], num_parameters + 1),
            )

        # if we have a mass matrix, we need to extend it
        def extend_mass_matrix(M):
            M_extend = [M.entries] * (num_parameters + 1)
            return pybamm.Matrix(block_diag(M_extend, format="csr"))

        model.mass_matrix = extend_mass_matrix(model.mass_matrix)

        if has_mass_matrix_inv:
            model.mass_matrix_inv = extend_mass_matrix(model.mass_matrix_inv)

    def _set_up_events(self, model, t_eval, inputs, vars_for_processing):
        # Check for heaviside and modulo functions in rhs and algebraic and add
        # discontinuity events if these exist.
        # Note: only checks for the case of t < X, t <= X, X < t, or X <= t,
        # but also accounts for the fact that t might be dimensional
        # Only do this for DAE models as ODE models can deal with discontinuities
        # fine

        if len(model.algebraic) > 0:
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

                    model.events.append(
                        pybamm.Event(
                            str(symbol),
                            expr,
                            pybamm.EventType.DISCONTINUITY,
                        )
                    )

                elif isinstance(symbol, pybamm.Modulo) and symbol.left == pybamm.t:
                    expr = symbol.right
                    num_events = 200 if (t_eval is None) else (t_eval[-1] // expr.value)

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
                init_sign = float(
                    np.sign(event.evaluate(0, model.y0.full(), inputs=inputs)).item()
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
            discontinuity_events,
        )

    def _set_consistent_initialization(self, model, time, inputs_dict):
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
        inputs_dict : dict
            Any input parameters to pass to the model when solving
        update_rhs : bool
            Whether to update the rhs. True for 'solve', False for 'step'.

        """

        if self._algebraic_solver or model.len_alg == 0:
            # Don't update model.y0
            return

        # Calculate consistent states for the algebraic equations
        model.y0 = self.calculate_consistent_state(model, time, inputs_dict)

    def calculate_consistent_state(self, model, time=0, inputs=None):
        """
        Calculate consistent state for the algebraic equations through
        root-finding. model.y0 is used as the initial guess for rootfinding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : float
            The time at which to calculate the initial conditions
        inputs: dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations). If self.root_method == None then returns
            model.y0.
        """
        pybamm.logger.debug("Start calculating consistent states")
        if self.root_method is None:
            return model.y0
        try:
            root_sol = self.root_method._integrate(model, np.array([time]), inputs)
        except pybamm.SolverError as e:
            raise pybamm.SolverError(
                f"Could not find consistent states: {e.args[0]}"
            ) from e
        pybamm.logger.debug("Found consistent states")

        self.check_extrapolation(root_sol, model.events)
        y0 = root_sol.all_ys[0]
        return y0

    def _solve_process_calculate_sensitivities_arg(
        inputs, model, calculate_sensitivities
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

    def solve(
        self,
        model,
        t_eval=None,
        inputs=None,
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
        inputs : dict or list, optional
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
        #
        # Argument "inputs" can be either a list of input dicts or
        # a single dict. The remaining of this function is only working
        # with variable "input_list", which is a list of dictionaries.
        # If "inputs" is a single dict, "inputs_list" is a list of only one dict.
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        model_inputs_list = [
            self._set_up_model_inputs(model, inputs) for inputs in inputs_list
        ]

        calculate_sensitivities_list, sensitivities_have_changed = (
            BaseSolver._solve_process_calculate_sensitivities_arg(
                model_inputs_list[0], model, calculate_sensitivities
            )
        )

        # (Re-)calculate consistent initialization
        # Assuming initial conditions do not depend on input parameters
        # when len(inputs_list) > 1, only `model_inputs_list[0]`
        # is passed to `_set_consistent_initialization`.
        # See https://github.com/pybamm-team/PyBaMM/pull/1261
        if len(inputs_list) > 1:
            all_inputs_names = set(
                itertools.chain.from_iterable(
                    [model_inputs.keys() for model_inputs in model_inputs_list]
                )
            )
            initial_conditions_node_names = set(
                [it.name for it in model.concatenated_initial_conditions.pre_order()]
            )
            if all_inputs_names.issubset(initial_conditions_node_names):
                raise pybamm.SolverError(
                    "Input parameters cannot appear in expression "
                    "for initial conditions."
                )

        # if any setup configuration has changed, we need to re-set up
        if sensitivities_have_changed:
            self._model_set_up.pop(model, None)
            # CasadiSolver caches its integrators using model, so delete this too
            if isinstance(self, pybamm.CasadiSolver):
                self.integrators.pop(model, None)

        # save sensitivity parameters so we can identify them later on
        # (FYI: this is used in the Solution class)
        model.calculate_sensitivities = calculate_sensitivities_list

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
            # It is assumed that when len(inputs_list) > 1, model set
            # up (initial condition, time-scale and length-scale) does
            # not depend on input parameters. Therefore, only `model_inputs[0]`
            # is passed to `set_up`.
            # See https://github.com/pybamm-team/PyBaMM/pull/1261
            self.set_up(model, model_inputs_list[0], t_eval)
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
                model.y0 = model.concatenated_initial_conditions.evaluate()
            else:
                # If the new initial conditions are different
                # and cannot be evaluated directly, set up again
                self.set_up(model, model_inputs_list[0], t_eval, ics_only=True)
            self._model_set_up[model]["initial conditions"] = (
                model.concatenated_initial_conditions
            )
        else:
            # Set the standard initial conditions
            self._set_initial_conditions(model, t_eval[0], model_inputs_list[0])

        # Solve for the consistent initialization
        self._set_consistent_initialization(model, t_eval[0], model_inputs_list[0])

        set_up_time = timer.time()
        timer.reset()

        # Check initial conditions don't violate events
        self._check_events_with_initialization(t_eval, model, model_inputs_list[0])

        # Process discontinuities
        (
            start_indices,
            end_indices,
            t_eval,
        ) = self._get_discontinuity_start_end_indices(model, inputs, t_eval)

        # Integrate separately over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a DAE)
        old_y0 = model.y0
        solutions = None
        for start_index, end_index in zip(start_indices, end_indices):
            pybamm.logger.verbose(
                f"Calling solver for {t_eval[start_index]} < t < {t_eval[end_index - 1]}"
            )
            if self.supports_parallel_solve:
                # Jax and IDAKLU solver can accept a list of inputs
                new_solutions = self._integrate(
                    model,
                    t_eval[start_index:end_index],
                    model_inputs_list,
                    t_interp,
                )
            else:
                ninputs = len(model_inputs_list)
                if ninputs == 1:
                    new_solution = self._integrate(
                        model,
                        t_eval[start_index:end_index],
                        model_inputs_list[0],
                        t_interp=t_interp,
                    )
                    new_solutions = [new_solution]
                else:
                    with mp.get_context(self._mp_context).Pool(processes=nproc) as p:
                        new_solutions = p.starmap(
                            self._integrate,
                            zip(
                                [model] * ninputs,
                                [t_eval[start_index:end_index]] * ninputs,
                                model_inputs_list,
                                [t_interp] * ninputs,
                            ),
                        )
                        p.close()
                        p.join()
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
                last_state = solutions[0].y[:, -1]
                # update y0 (for DAE solvers, this updates the initial guess for the
                # rootfinder)
                model.y0 = last_state
                if len(model.algebraic) > 0:
                    model.y0 = self.calculate_consistent_state(
                        model, t_eval[end_index], model_inputs_list[0]
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
        model.y0 = old_y0

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
    def _get_discontinuity_start_end_indices(model, inputs, t_eval):
        if not model.discontinuity_events_eval:
            pybamm.logger.verbose("No discontinuity events found")
            return [0], [len(t_eval)], t_eval

        # Calculate discontinuities
        discontinuities = [
            # Assuming that discontinuities do not depend on
            # input parameters when len(input_list) > 1, only
            # `inputs` is passed to `evaluate`.
            # See https://github.com/pybamm-team/PyBaMM/pull/1261
            event.expression.evaluate(inputs=inputs)
            for event in model.discontinuity_events_eval
        ]

        # make sure they are increasing in time
        discontinuities = sorted(discontinuities)

        # remove any identical discontinuities
        discontinuities = [
            v
            for i, v in enumerate(discontinuities)
            if (
                i == len(discontinuities) - 1
                or discontinuities[i] < discontinuities[i + 1]
            )
            and v > 0
        ]

        # remove any discontinuities after end of t_eval
        discontinuities = [v for v in discontinuities if v < t_eval[-1]]

        pybamm.logger.verbose(f"Discontinuity events found at t = {discontinuities}")
        if isinstance(inputs, list):
            raise pybamm.SolverError(
                "Cannot solve for a list of input parameters"
                " sets with discontinuities"
            )

        # insert time points around discontinuities in t_eval
        # keep track of subsections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        eps = sys.float_info.epsilon
        for dtime in discontinuities:
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
    def _check_events_with_initialization(t_eval, model, inputs_dict):
        num_terminate_events = len(model.terminate_events_eval)
        if num_terminate_events == 0:
            return

        if model.convert_to_format == "casadi":
            inputs = casadi.vertcat(*[x for x in inputs_dict.values()])

        events_eval = [None] * num_terminate_events
        for idx, event in enumerate(model.terminate_events_eval):
            if model.convert_to_format == "casadi":
                event_eval = event(t_eval[0], model.y0, inputs)
            elif model.convert_to_format in ["python", "jax"]:
                event_eval = event(t=t_eval[0], y=model.y0, inputs=inputs_dict)
            events_eval[idx] = event_eval

        events_eval = np.array(events_eval)
        if any(events_eval < 0):
            # find the events that were triggered by initial conditions
            termination_events = [
                x for x in model.events if x.event_type == pybamm.EventType.TERMINATION
            ]
            idxs = np.where(events_eval < 0)[0]
            event_names = [termination_events[idx].name for idx in idxs]
            raise pybamm.SolverError(
                f"Events {event_names} are non-positive at initial conditions"
            )

    def _set_sens_initial_conditions_from(
        self, solution: pybamm.Solution, model: pybamm.BaseModel
    ) -> tuple:
        """
        A restricted version of BaseModel.set_initial_conditions_from that only extracts the
        sensitivities from a solution object, and only for a model that has been descretised.
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
            casadi.vertcat(*[eq for _, eq in sorted(zip(slices, init))])
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
            so are not guarenteed to be within the tolerances set by the solver, please raise an issue if you
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
        elif t_eval[0] != 0 or t_eval[-1] != dt:
            raise pybamm.SolverError(
                "Elements inside array t_eval must lie in the closed interval 0 to dt"
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
        calculate_sensitivities_list, sensitivities_have_changed = (
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
            model.y0 = last_state.all_ys[0]
            if using_sensitivities and isinstance(last_state._all_sensitivities, dict):
                full_sens = last_state._all_sensitivities["all"][0]
                model.y0S = tuple(full_sens[:, i] for i in range(full_sens.shape[1]))

        else:
            _, concatenated_initial_conditions = model.set_initial_conditions_from(
                old_solution, return_type="ics"
            )
            model.y0 = concatenated_initial_conditions.evaluate(0, inputs=model_inputs)
            if using_sensitivities:
                model.y0S = self._set_sens_initial_conditions_from(old_solution, model)

        # hopefully we'll get rid of explicit sensitivities soon so we can remove this
        explicit_sensitivities = model.len_rhs_sens > 0 or model.len_alg_sens > 0
        if (
            explicit_sensitivities
            and using_sensitivities
            and not isinstance(old_solution, pybamm.EmptySolution)
            and not old_solution.all_models[-1] == model
        ):
            y0_list = []
            if model.len_rhs > 0:
                y0_list.append(model.y0[: model.len_rhs])
                for s in model.y0S:
                    y0_list.append(s[: model.len_rhs])
            if model.len_alg > 0:
                y0_list.append(model.y0[model.len_rhs :])
                for s in model.y0S:
                    y0_list.append(s[model.len_rhs :])
            model.y0 = casadi.vertcat(*y0_list)

        set_up_time = timer.time()

        # (Re-)calculate consistent initialization
        self._set_consistent_initialization(model, t_start_shifted, model_inputs)

        # Check consistent initialization doesn't violate events
        self._check_events_with_initialization(t_eval, model, model_inputs)

        # Step
        pybamm.logger.verbose(f"Stepping for {t_start_shifted:.0f} < t < {t_end:.0f}")
        timer.reset()

        # API for _integrate is different for JaxSolver and IDAKLUSolver
        if self.supports_parallel_solve:
            solutions = self._integrate(model, t_eval, [model_inputs], t_interp)
            solution = solutions[0]
        else:
            solution = self._integrate(model, t_eval, model_inputs, t_interp)
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
        if solution.termination == "final time":
            return (
                solution,
                "the solver successfully reached the end of the integration interval",
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

        if self._on_extrapolation == "error":
            raise pybamm.SolverError(
                "Solver failed because the following "
                f"interpolation bounds were exceeded: {extrap_events}. "
                "You may need to provide additional interpolation points "
                "outside these bounds."
            )
        elif self._on_extrapolation == "warn":
            name = solution.all_models[-1].name
            warnings.warn(
                f"While solving {name} extrapolation occurred " f"for {extrap_events}",
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
        if system_type.lower() in ["linux", "darwin"]:
            return "fork"
        return "spawn"

    @staticmethod
    def _set_up_model_inputs(model, inputs):
        """Set up input parameters"""
        if inputs is None:
            inputs = {}
        else:
            inputs = ParameterValues.check_parameter_values(inputs)

        # Go through all input parameters that can be found in the model
        # Only keep the ones that are actually used in the model
        # If any of them are *not* provided by "inputs", raise an error
        inputs_in_model = {}
        for input_param in model.input_parameters:
            name = input_param.name
            if name not in inputs:
                raise pybamm.SolverError(f"No value provided for input '{name}'")
            inputs_in_model[name] = inputs[name]

        inputs = inputs_in_model

        ordered_inputs_names = list(inputs.keys())
        ordered_inputs_names.sort()
        ordered_inputs = {name: inputs[name] for name in ordered_inputs_names}

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
        y_and_S = vars_for_processing["y_and_S"]
        p_casadi_stacked = vars_for_processing["p_casadi_stacked"]
        calculate_sensitivities_explicit = vars_for_processing[
            "calculate_sensitivities_explicit"
        ]
        # Process with CasADi
        report(f"Converting {name} to CasADi")
        casadi_expression = symbol.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
        # Add sensitivity vectors to the rhs and algebraic equations
        jacp = None
        if calculate_sensitivities_explicit:
            # The formulation is as per Park, S., Kato, D., Gima, Z., Klein, R.,
            # & Moura, S. (2018).  Optimal experimental design for
            # parameterization of an electrochemical lithium-ion battery model.
            # Journal of The Electrochemical Society, 165(7), A1309.". See #1100
            # for details
            pS_casadi_stacked = vars_for_processing["pS_casadi_stacked"]
            y_diff = vars_for_processing["y_diff"]
            y_alg = vars_for_processing["y_alg"]
            S_x = vars_for_processing["S_x"]
            S_z = vars_for_processing["S_z"]

            if name == "RHS" and model.len_rhs > 0:
                report(
                    "Creating explicit forward sensitivity equations "
                    "for rhs using CasADi"
                )
                df_dx = casadi.jacobian(casadi_expression, y_diff)
                df_dp = casadi.jacobian(casadi_expression, pS_casadi_stacked)
                S_x_mat = S_x.reshape((model.len_rhs, pS_casadi_stacked.shape[0]))
                if model.len_alg == 0:
                    S_rhs = (df_dx @ S_x_mat + df_dp).reshape((-1, 1))
                else:
                    df_dz = casadi.jacobian(casadi_expression, y_alg)
                    S_z_mat = S_z.reshape((model.len_alg, pS_casadi_stacked.shape[0]))
                    S_rhs = (df_dx @ S_x_mat + df_dz @ S_z_mat + df_dp).reshape((-1, 1))
                casadi_expression = casadi.vertcat(casadi_expression, S_rhs)
            if name == "algebraic" and model.len_alg > 0:
                report(
                    "Creating explicit forward sensitivity equations "
                    "for algebraic using CasADi"
                )
                dg_dz = casadi.jacobian(casadi_expression, y_alg)
                dg_dp = casadi.jacobian(casadi_expression, pS_casadi_stacked)
                S_z_mat = S_z.reshape((model.len_alg, pS_casadi_stacked.shape[0]))
                if model.len_rhs == 0:
                    S_alg = (dg_dz @ S_z_mat + dg_dp).reshape((-1, 1))
                else:
                    dg_dx = casadi.jacobian(casadi_expression, y_diff)
                    S_x_mat = S_x.reshape((model.len_rhs, pS_casadi_stacked.shape[0]))
                    S_alg = (dg_dx @ S_x_mat + dg_dz @ S_z_mat + dg_dp).reshape((-1, 1))
                casadi_expression = casadi.vertcat(casadi_expression, S_alg)
            if name == "initial_conditions":
                if model.len_rhs == 0 or model.len_alg == 0:
                    S_0 = casadi.jacobian(casadi_expression, pS_casadi_stacked).reshape(
                        (-1, 1)
                    )
                    casadi_expression = casadi.vertcat(casadi_expression, S_0)
                else:
                    x0 = casadi_expression[: model.len_rhs]
                    z0 = casadi_expression[model.len_rhs :]
                    Sx_0 = casadi.jacobian(x0, pS_casadi_stacked).reshape((-1, 1))
                    Sz_0 = casadi.jacobian(z0, pS_casadi_stacked).reshape((-1, 1))
                    casadi_expression = casadi.vertcat(x0, Sx_0, z0, Sz_0)
        elif model.calculate_sensitivities:
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
                    [t_casadi, y_and_S, p_casadi_stacked],
                    [
                        casadi.densify(
                            casadi.jacobian(casadi_expression, p_casadi[pname])
                        )
                        for pname in model.calculate_sensitivities
                    ],
                )

        if use_jacobian:
            report(f"Calculating jacobian for {name} using CasADi")
            jac_casadi = casadi.jacobian(casadi_expression, y_and_S)
            jac = casadi.Function(
                name + "_jac",
                [t_casadi, y_and_S, p_casadi_stacked],
                [jac_casadi],
            )

            v = casadi.MX.sym(
                "v",
                model.len_rhs_and_alg + model.len_rhs_sens + model.len_alg_sens,
            )
            jac_action_casadi = casadi.densify(
                casadi.jtimes(casadi_expression, y_and_S, v)
            )
            jac_action = casadi.Function(
                name + "_jac_action",
                [t_casadi, y_and_S, p_casadi_stacked, v],
                [jac_action_casadi],
            )
        else:
            jac = None
            jac_action = None

        func = casadi.Function(
            name, [t_casadi, y_and_S, p_casadi_stacked], [casadi_expression]
        )

    return func, jac, jacp, jac_action
