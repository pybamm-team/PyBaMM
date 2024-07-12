import copy
import itertools
import numbers
import sys
import warnings
import platform

# import asyncio
import multiprocessing as mp


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
    options: dict, optional
        List of options, defaults are:
        options = {
            # Number of threads available for OpenMP
            "num_threads": 1,
        }
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
        options=None,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.root_tol = root_tol
        self.root_method = root_method
        self.extrap_tol = extrap_tol or -1e-10
        self.output_variables = [] if output_variables is None else output_variables
        self._model_set_up = {}
        default_options = {
            "num_threads": 1,
        }
        if options is None:
            options = default_options
        else:
            for key, value in default_options.items():
                if key not in options:
                    options[key] = value
        self._base_options = options

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self.ode_solver = False
        self.algebraic_solver = False
        self._on_extrapolation = "warn"
        self.computed_var_fcns = {}
        self._mp_context = self.get_platform_context(platform.system())

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

    def set_up(self, model, inputs=None, t_eval=None, ics_only=False, batch_size=1):
        """Unpack model, perform checks, and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict or list of dict, optional
            Any input parameters to pass to the model when solving
        t_eval : numeric type, optional
            The times (in seconds) at which to compute the solution
        """

        if inputs is None:
            inputs = [{}]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if ics_only:
            pybamm.logger.info("Start solver set-up, initial_conditions only")
        else:
            pybamm.logger.info("Start solver set-up")

        self._check_and_prepare_model_inplace(model)

        # set default calculate sensitivities on model
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []

        # see if we need to form the explicit sensitivity equations
        calculate_sensitivities_explicit = False
        if model.calculate_sensitivities and not isinstance(self, pybamm.IDAKLUSolver):
            calculate_sensitivities_explicit = True

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
            inputs,
            batch_size=batch_size,
            nthreads=self._base_options["num_threads"],
            use_jacobian=False,
        )
        model.initial_conditions_eval = initial_conditions
        model.jacp_initial_conditions_eval = jacp_ic

        # evaluate initial condition
        y0_total_size = batch_size * (
            model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
        )
        y_zero = np.zeros((y0_total_size, 1))

        # todo: vry wasteful, but can't pass in batched_inputs from upper scope as this would change the signature
        batched_inputs = [
            self._inputs_to_stacked_vect(
                inputs[i * batch_size : (i + 1) * batch_size], model.convert_to_format
            )
            for i in range(len(inputs) // batch_size)
        ]

        model.y0_list = [
            initial_conditions(0.0, y_zero, inputs) for inputs in batched_inputs
        ]

        if jacp_ic is None:
            model.y0S_list = []
        else:
            model.y0S_list = [
                jacp_ic(0.0, y_zero, stacked_inputs)
                for stacked_inputs in batched_inputs
            ]

        if ics_only:
            pybamm.logger.info("Finish solver set-up")
            return

        # Process rhs, algebraic, residual and event expressions
        # and wrap in callables
        is_casadi_solver = isinstance(
            self.root_method, pybamm.CasadiAlgebraicSolver
        ) or isinstance(self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver))
        if is_casadi_solver and len(model.rhs) > 0:
            rhs = model.mass_matrix_inv @ model.concatenated_rhs
        else:
            rhs = model.concatenated_rhs
        rhs, jac_rhs, jacp_rhs, jac_rhs_action = process(
            rhs,
            "RHS",
            vars_for_processing,
            inputs,
            batch_size=batch_size,
            nthreads=self._base_options["num_threads"],
        )

        algebraic, jac_algebraic, jacp_algebraic, jac_algebraic_action = process(
            model.concatenated_algebraic,
            "algebraic",
            vars_for_processing,
            inputs,
            batch_size=batch_size,
            nthreads=self._base_options["num_threads"],
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
        ) = process(
            rhs_algebraic,
            "rhs_algebraic",
            vars_for_processing,
            inputs,
            batch_size=batch_size,
            nthreads=self._base_options["num_threads"],
        )

        (
            casadi_switch_events,
            terminate_events,
            interpolant_extrapolation_events,
            discontinuity_events,
        ) = self._set_up_events(model, t_eval, inputs, vars_for_processing, batch_size)

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
        if is_casadi_solver:
            # can use DAE solver to solve model with algebraic equations only
            # todo: do I need this check?
            if len(model.rhs) > 0:
                model.casadi_rhs = rhs
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
                if isinstance(
                    model.variables_and_events[key], pybamm.ExplicitTimeIntegral
                ):
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
                    inputs,
                    batch_size=batch_size,
                    nthreads=self._base_options["num_threads"],
                    use_jacobian=True,
                    return_jacp_stacked=True,
                )

        pybamm.logger.info("Finish solver set-up")

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
        if self.algebraic_solver is True and len(model.rhs) > 0:
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
            for name, value in inputs[0].items():
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
                if isinstance(inputs[0][name], numbers.Number):
                    num_parameters += 1
                else:
                    num_parameters += len(inputs[0][name])
            num_parameters *= len(inputs)
            model.len_rhs_sens = model.len_rhs * num_parameters
            model.len_alg_sens = model.len_alg * num_parameters
        else:
            model.len_rhs_sens = 0
            model.len_alg_sens = 0

        # if we will change the equations to include the explicit sensitivity
        # equations, then we also need to update the mass matrix and bounds.
        # First, we reset the mass matrix and bounds back to their original form
        # if they have been extended
        if model.bounds[0].shape[0] > model.len_rhs_and_alg:
            model.bounds = (
                model.bounds[0][: model.len_rhs_and_alg],
                model.bounds[1][: model.len_rhs_and_alg],
            )

        # now we can extend them by the number of sensitivity parameters
        # if needed
        if calculate_sensitivities_explicit:
            if model.len_rhs != 0:
                n_inputs = model.len_rhs_sens // model.len_rhs
            elif model.len_alg != 0:
                n_inputs = model.len_alg_sens // model.len_alg
            if model.bounds[0].shape[0] == model.len_rhs_and_alg:
                model.bounds = (
                    np.repeat(model.bounds[0], n_inputs + 1),
                    np.repeat(model.bounds[1], n_inputs + 1),
                )

    def _set_up_events(self, model, t_eval, inputs, vars_for_processing, batch_size):
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
                    expr = None
                    if symbol.right == pybamm.t:
                        expr = symbol.left
                    else:
                        if symbol.left == pybamm.t:
                            expr = symbol.right

                    # Update the events if the heaviside function depended on t
                    if expr is not None:
                        model.events.append(
                            pybamm.Event(
                                str(symbol),
                                expr,
                                pybamm.EventType.DISCONTINUITY,
                            )
                        )
                elif isinstance(symbol, pybamm.Modulo):
                    if symbol.left == pybamm.t:
                        expr = symbol.right
                        num_events = 200
                        if t_eval is not None:
                            num_events = t_eval[-1] // expr.value

                        for i in np.arange(num_events):
                            model.events.append(
                                pybamm.Event(
                                    str(symbol),
                                    expr * pybamm.Scalar(i + 1),
                                    pybamm.EventType.DISCONTINUITY,
                                )
                            )
                else:
                    pass

        casadi_switch_events = []
        terminate_events = []
        interpolant_extrapolation_events = []
        discontinuity_events = []
        for n, event in enumerate(model.events):
            if event.event_type == pybamm.EventType.DISCONTINUITY:
                # discontinuity events are evaluated before the solver is called,
                # so don't need to process them
                discontinuity_events.append(event)
            elif event.event_type == pybamm.EventType.SWITCH:
                if (
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
                    # note: assumes that sign for all batches is the same
                    init_sign = float(
                        np.sign(
                            event.evaluate(0, model.y0_list[0].full(), inputs=inputs)
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
                        inputs,
                        batch_size=batch_size,
                        nthreads=self._base_options["num_threads"],
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
                    inputs,
                    batch_size=batch_size,
                    nthreads=self._base_options["num_threads"],
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

    def _set_initial_conditions(
        self, model, time, inputs_list, update_rhs, batched_inputs
    ):
        """
        Set initial conditions for the model. This is skipped if the solver is an
        algebraic solver (since this would make the algebraic solver redundant), and if
        the model doesn't have any algebraic equations (since there are no initial
        conditions to be calculated in this case).

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        inputs_list: list of dict
            Any input parameters to pass to the model when solving
        update_rhs : bool
            Whether to update the rhs. True for 'solve', False for 'step'.

        """

        y0_total_size = model.y0_list[0].shape[0]
        y_zero = np.zeros((y0_total_size, 1))

        if self.algebraic_solver is True:
            # Don't update model.y0_list
            return
        elif len(model.algebraic) == 0:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                y0_list = [
                    model.initial_conditions_eval(time, y_zero, inputs)
                    for inputs in batched_inputs
                ]
            else:
                # Don't update model.y0_list
                return
        else:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                y0_from_inputs = [
                    model.initial_conditions_eval(time, y_zero, inputs)
                    for inputs in batched_inputs
                ]
                # Reuse old solution for algebraic equations
                y0_from_model = model.y0_list
                len_rhs = model.len_rhs + model.len_rhs_sens
                # update model.y0_list, which is used for initialising the algebraic solver
                if len_rhs == 0:
                    model.y0_list = y0_from_model
                else:
                    for i in range(len(y0_from_inputs)):
                        _, y_alg = self._unzip_state_vector(model, y0_from_model[i])
                        y_diff, _ = self._unzip_state_vector(model, y0_from_inputs[i])
                        model.y0_list[i] = self._zip_state_vector(model, y_diff, y_alg)
            y0_list = self.calculate_consistent_state(model, time, inputs_list)

            # concatenate batches again
            nbatches = len(batched_inputs)
            batch_size = len(inputs_list) // nbatches
            y0_list_of_list = [
                y0_list[i * batch_size : (i + 1) * batch_size] for i in range(nbatches)
            ]
            if isinstance(y0_list[0], casadi.DM):
                y0_list = [casadi.vertcat(*y0s) for y0s in y0_list_of_list]
            else:
                y0_list = [np.vstack(y0s) for y0s in y0_list_of_list]

        # Make y0 a function of inputs if doing symbolic with casadi
        model.y0_list = y0_list

    def calculate_consistent_state(self, model, time=0, inputs=None):
        """
        Calculate consistent state for the algebraic equations through
        root-finding. model.y0_list is used as the initial guess for rootfinding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : float
            The time at which to calculate the states
        inputs: list of dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        y0_consistent : list of array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations). If self.root_method == None then returns
            model.y0_list.
        """
        pybamm.logger.debug("Start calculating consistent states")
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
    def _handle_integrate_defaults(
        model: pybamm.BaseModel,
        inputs_list: list[dict] | None,
        batched_inputs: list | None,
    ) -> tuple[list[dict], list, int, int, list]:
        """
        convenience function to handle the default inputs for self._integrate

        Returns
        -------
        inputs_list : list of dict
            The list of inputs to pass to the model when solving
        batched_inputs : list of array
            batched inputs parameters in list of array form
        nbatches : int
            The number of batches to solve
        batch_size : int
            The size of each batch
        y0S_list : list
            The list of initial conditions for the sensitivities
        """
        inputs_list = inputs_list or [{}]

        if batched_inputs is not None:
            nbatches = len(batched_inputs)
            batch_size = len(inputs_list) // nbatches
        else:
            batch_size = model.batch_size
            nbatches = len(inputs_list) // batch_size

        # check if we need to recalculate batched_inputs
        if batched_inputs is None:
            batched_inputs = [
                BaseSolver._inputs_to_stacked_vect(
                    inputs_list[i * batch_size : (i + 1) * batch_size],
                    model.convert_to_format,
                )
                for i in range(len(inputs_list) // batch_size)
            ]

        if not hasattr(model, "y0S_list") or len(model.y0S_list) == 0:
            y0S_list = [None] * nbatches
        else:
            y0S_list = model.y0S_list
        return inputs_list, batched_inputs, nbatches, batch_size, y0S_list

    def _integrate(self, model, t_eval, inputs_list=None, batched_inputs=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_list : list of dict
            Any input parameters to pass to the model when solving
        batched_inputs : list of array
            batched inputs parameters in list of array form
        """
        inputs_list, batched_inputs, nbatches, batch_size, y0S_list = (
            self._handle_integrate_defaults(model, inputs_list, batched_inputs)
        )

        if nbatches == 1:
            return self._integrate_batch(
                model,
                t_eval,
                model.y0_list[0],
                y0S_list[0],
                inputs_list,
                batched_inputs[0],
            )

        # async io is not parallel, but if solve is io bound, it can be faster
        # async def solve_model_batches():
        #    async def solve_model_async(y0, y0S, inputs, inputs_array):
        #        return self._integrate_batch(
        #            model, t_eval, y0, y0S, inputs, inputs_array
        #        )

        #    coro = []
        #    for i in range(nbatches):
        #        coro.append(
        #            asyncio.create_task(
        #                solve_model_async(
        #                    model.y0_list[i],
        #                    y0S_list[i],
        #                    inputs_list[i * batch_size : (i + 1) * batch_size],
        #                    batched_inputs[i],
        #                )
        #            )
        #        )
        #    return await asyncio.gather(*coro)

        # new_solutions = asyncio.run(solve_model_batches())

        # new_solutions = []
        # for i in range(nbatches):
        #    new_solutions.append(self._integrate_batch(model, t_eval, model.y0_list[i], y0S_list[i], inputs_list[i * batch_size : (i + 1) * batch_size], batched_inputs[i]))

        threads_per_batch = max(self._base_options["num_threads"] // nbatches, 1)
        nproc = self._base_options["num_threads"] // threads_per_batch
        with mp.get_context(self._mp_context).Pool(processes=nproc) as p:
            model_list = [model] * nbatches
            t_eval_list = [t_eval] * nbatches
            y0_list = model.y0_list
            inputs_list_of_list = [
                inputs_list[i * batch_size : (i + 1) * batch_size]
                for i in range(nbatches)
            ]
            new_solutions = p.starmap(
                self._integrate_batch,
                zip(
                    model_list,
                    t_eval_list,
                    y0_list,
                    y0S_list,
                    inputs_list_of_list,
                    batched_inputs,
                ),
            )
            p.close()
            p.join()
        new_solutions_flat = [sol for sublist in new_solutions for sol in sublist]
        return new_solutions_flat

    def _integrate_batch(self, model, t_eval, y0, y0S, inputs_list, inputs):
        """
        Solve a single batch for the DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : numeric type
            The times at which to compute the solution
        inputs_list : list of dict, optional
            Any input parameters to pass to the model when solving
        inputs : array, optional
            The input parameters in array form, to pass to the model when solving
        """
        raise NotImplementedError

    def solve(
        self,
        model,
        t_eval=None,
        inputs=None,
        batch_size=1,
        calculate_sensitivities=False,
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
        batch_size: int, optional
            If `n_i` sets of inputs are provided, the solver will batch the solves in
            groups of `batch_size` when solving. Each batch is solved as a set of `n_s * batch_size`
            equations, where `n_s` is the number of equations in the model. Defaults to 1.
        calculate_sensitivities : list of str or bool, optional
            Whether the solver calculates sensitivities of all input parameters. Defaults to False.
            If only a subset of sensitivities are required, can also pass a
            list of input parameter names

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

        # check that len(inputs) is a multiple of batch_size
        if isinstance(inputs, list) and len(inputs) % batch_size != 0:
            raise ValueError(
                "len(inputs) must be a multiple of batch_size, but "
                f"len(inputs) = {len(inputs)} and batch_size = {batch_size}"
            )

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
                # check for a discretised model without original parameters
                if not (
                    model.concatenated_rhs is not None
                    or model.concatenated_algebraic is not None
                ):
                    raise pybamm.ModelError(
                        "Cannot solve empty model, use `pybamm.DummySolver` instead"
                    )

        # t_eval can only be None if the solver is an algebraic solver. In that case
        # set it to 0
        if t_eval is None:
            if self.algebraic_solver is True:
                t_eval = np.array([0])
            else:
                raise ValueError("t_eval cannot be None")

        # If t_eval is provided as [t0, tf] return the solution at 100 points
        elif isinstance(t_eval, list):
            if len(t_eval) == 1 and self.algebraic_solver is True:
                t_eval = np.array(t_eval)
            elif len(t_eval) != 2:
                raise pybamm.SolverError(
                    "'t_eval' can be provided as an array of times at which to "
                    "return the solution, or as a list [t0, tf] where t0 is the "
                    "initial time and tf is the final time, but has been provided "
                    f"as a list of length {len(t_eval)}."
                )
            else:
                t_eval = np.linspace(t_eval[0], t_eval[-1], 100)

        # Make sure t_eval is monotonic
        if (np.diff(t_eval) < 0).any():
            raise pybamm.SolverError("t_eval must increase monotonically")

        # Set up inputs
        #
        # Argument "inputs" can be either a list of input dicts or
        # a single dict. The remaining of this function is only working
        # with variable "input_list", which is a list of dictionaries.
        # If "inputs" is a single dict, "inputs_list" is a list of only one dict.
        if isinstance(inputs, dict):
            inputs_list = [inputs]
        else:
            inputs_list = inputs or [{}]
        model_inputs_list = [
            self._set_up_model_inputs(model, inputs) for inputs in inputs_list
        ]

        # get a list-only version of calculate_sensitivities
        if isinstance(calculate_sensitivities, bool):
            if calculate_sensitivities:
                calculate_sensitivities_list = [p for p in inputs_list[0].keys()]
            else:
                calculate_sensitivities_list = []
        else:
            calculate_sensitivities_list = calculate_sensitivities

        # Check that calculate_sensitivites or batch size have not been updated
        calculate_sensitivities_list.sort()
        if not hasattr(model, "calculate_sensitivities"):
            model.calculate_sensitivities = []
        if not hasattr(model, "batch_size"):
            model.batch_size = batch_size

        model.calculate_sensitivities.sort()
        sensitivities_changed = (
            calculate_sensitivities_list != model.calculate_sensitivities
        )
        batch_size_changed = model.batch_size != batch_size

        if sensitivities_changed or batch_size_changed:
            self._model_set_up.pop(model, None)
            # CasadiSolver caches its integrators using model, so delete this too
            if isinstance(self, pybamm.CasadiSolver):
                self.integrators.pop(model, None)

        # save sensitivity parameters and batch size so we can identify them later on
        # (FYI: this is used in the Solution class)
        model.calculate_sensitivities = calculate_sensitivities_list
        model.batch_size = batch_size

        # Set up (if not done already)
        timer = pybamm.Timer()
        if model not in self._model_set_up:
            if len(self._model_set_up) > 0:
                existing_model = next(iter(self._model_set_up))
                raise RuntimeError(
                    "This solver has already been initialised for model "
                    f'"{existing_model.name}". Please create a separate '
                    "solver for this model"
                )
            self.set_up(model, model_inputs_list, t_eval, batch_size=batch_size)
            self._model_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )
        else:
            ics_set_up = self._model_set_up[model]["initial conditions"]
            # Check that initial conditions have not been updated
            if ics_set_up != model.concatenated_initial_conditions:
                if self.algebraic_solver is True:
                    # For an algebraic solver, we don't need to set up the initial
                    # conditions function and we can just evaluate
                    # model.concatenated_initial_conditions
                    model.y0_list = [
                        model.concatenated_initial_conditions.evaluate()
                    ] * (len(inputs_list) // batch_size)
                else:
                    # If the new initial conditions are different
                    # and cannot be evaluated directly, set up again
                    self.set_up(
                        model,
                        model_inputs_list,
                        t_eval,
                        ics_only=True,
                        batch_size=batch_size,
                    )
                self._model_set_up[model]["initial conditions"] = (
                    model.concatenated_initial_conditions
                )

        set_up_time = timer.time()
        timer.reset()

        batched_inputs = [
            self._inputs_to_stacked_vect(
                model_inputs_list[i * batch_size : (i + 1) * batch_size],
                model.convert_to_format,
            )
            for i in range(len(inputs_list) // batch_size)
        ]

        self._set_initial_conditions(
            model,
            t_eval[0],
            model_inputs_list,
            update_rhs=True,
            batched_inputs=batched_inputs,
        )

        # Check initial conditions don't violate events
        for y0, inpts in zip(model.y0_list, batched_inputs):
            self._check_events_with_initial_conditions(t_eval, model, y0, inpts)

        # Process discontinuities
        (
            start_indices,
            end_indices,
            t_eval,
        ) = self._get_discontinuity_start_end_indices(model, inputs, t_eval)

        # Integrate separately over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a DAE)
        old_y0_list = model.y0_list
        solutions = None
        for start_index, end_index in zip(start_indices, end_indices):
            pybamm.logger.verbose(
                f"Calling solver for {t_eval[start_index]} < t < {t_eval[end_index - 1]}"
            )
            new_solutions = self._integrate(
                model,
                t_eval[start_index:end_index],
                model_inputs_list,
                batched_inputs,
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
                last_state = solutions[0].y[:, -1]
                # update y0 (for DAE solvers, this updates the initial guess for the
                # rootfinder)
                model.y0_list = last_state
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
            self.algebraic_solver is False
            and len(solutions[0].all_ts) == 1
            and len(solutions[0].all_ts[0]) == 1
        ):
            raise pybamm.SolverError(
                "Solution time vector has length 1. "
                "Check whether simulation terminated too early."
            )

        # Return solution(s)
        if len(inputs_list) == 1:
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
    def _check_events_with_initial_conditions(t_eval, model, y0, inputs):
        num_terminate_events = len(model.terminate_events_eval)
        if num_terminate_events == 0:
            return

        events_eval = [None] * num_terminate_events
        for idx, event in enumerate(model.terminate_events_eval):
            event_eval = event(t_eval[0], y0, inputs)
            events_eval[idx] = event_eval

        if model.convert_to_format == "casadi":
            events_eval = casadi.vertcat(*events_eval).toarray().flatten()
        else:
            events_eval = np.vstack(events_eval).flatten()
        if any(events_eval < 0):
            # find the events that were triggered by initial conditions
            termination_events = [
                x for x in model.events if x.event_type == pybamm.EventType.TERMINATION
            ]
            idxs = np.where(events_eval < 0)[0]
            event_names = [termination_events[idx / len(inputs)].name for idx in idxs]
            raise pybamm.SolverError(
                f"Events {event_names} are non-positive at initial conditions"
            )

    def step(
        self,
        old_solution,
        model,
        dt,
        t_eval=None,
        npts=None,
        inputs=None,
        save=True,
    ):
        """
        Step the solution of the model forward by a given time increment. The
        first time this method is called it executes the necessary setup by
        calling `self.set_up(model)`.

        Parameters
        ----------
        old_solution : :class:`pybamm.Solution` or list of :class:`pybamm.Solution` or None
            The previous solution to be added to. If `None`, a new solution is created.
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        dt : numeric type
            The timestep (in seconds) over which to step the solution
        t_eval : list or numpy.ndarray, optional
            An array of times at which to return the solution during the step
            (Note: t_eval is the time measured from the start of the step, so should start at 0 and end at dt).
            By default, the solution is returned at t0 and t0 + dt.
        npts : deprecated
        inputs : dict or list of dict, optional
            Any input parameters to pass to the model when solving
        save : bool, optional
            Save solution with all previous timesteps. Defaults to True.
        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

        """
        # restrict batch_size to 1 for now
        batch_size = 1
        model.batch_size = batch_size

        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        if old_solution is None:
            old_solutions = [pybamm.EmptySolution()] * len(inputs_list)
        elif not isinstance(old_solution, list):
            old_solutions = [old_solution]

        if not (
            isinstance(old_solutions[0], pybamm.EmptySolution)
            or old_solutions[0].termination == "final time"
            or "[experiment]" in old_solutions[0].termination
        ):
            # Return same solution as an event has already been triggered
            # With hack to allow stepping past experiment current / voltage cut-off
            if len(old_solutions) == 1:
                return old_solutions[0]
            else:
                return old_solutions

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
                raise pybamm.ModelError(
                    "Cannot step empty model, use `pybamm.DummySolver` instead"
                )

        # Make sure dt is greater than the offset
        step_start_offset = pybamm.settings.step_start_offset
        if dt <= step_start_offset:
            raise pybamm.SolverError(
                f"Step time must be at least {pybamm.TimerTime(step_start_offset)}"
            )

        # Raise deprecation warning for npts and convert it to t_eval
        if npts is not None:
            warnings.warn(
                "The 'npts' parameter is deprecated, use 't_eval' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            t_eval = np.linspace(0, dt, npts)

        if t_eval is not None:
            # Checking if t_eval lies within range
            if t_eval[0] != 0 or t_eval[-1] != dt:
                raise pybamm.SolverError(
                    "Elements inside array t_eval must lie in the closed interval 0 to dt"
                )

        else:
            t_eval = np.array([0, dt])

        t_start = old_solutions[0].t[-1]
        t_eval = t_start + t_eval
        t_end = t_start + dt

        if t_start == 0:
            t_start_shifted = t_start
        else:
            # offset t_start by t_start_offset (default 1 ns)
            # to avoid repeated times in the solution
            # from having the same time at the end of the previous step and
            # the start of the next step
            t_start_shifted = t_start + step_start_offset
            t_eval[0] = t_start_shifted

        # Set timer
        timer = pybamm.Timer()

        # Set up inputs
        model_inputs_list = [
            self._set_up_model_inputs(model, inputs) for inputs in inputs_list
        ]

        first_step_this_model = False
        if model not in self._model_set_up:
            first_step_this_model = True
            if len(self._model_set_up) > 0:
                existing_model = next(iter(self._model_set_up))
                raise RuntimeError(
                    "This solver has already been initialised for model "
                    f'"{existing_model.name}". Please create a separate '
                    "solver for this model"
                )
            self.set_up(model, model_inputs_list)
            self._model_set_up.update(
                {model: {"initial conditions": model.concatenated_initial_conditions}}
            )

        if (
            isinstance(old_solutions[0], pybamm.EmptySolution)
            and old_solutions[0].termination is None
        ):
            pybamm.logger.verbose(f"Start stepping {model.name} with {self.name}")

        if isinstance(old_solutions[0], pybamm.EmptySolution):
            if not first_step_this_model:
                # reset y0 to original initial conditions
                self.set_up(model, model_inputs_list, ics_only=True)
        else:
            if old_solutions[0].all_models[-1] == model:
                # initialize with old solution
                y0s = [s.all_ys[-1][:, -1] for s in old_solutions]

            else:
                y0s = []
                for soln, inputs in zip(old_solutions, model_inputs_list):
                    _, concatenated_initial_conditions = (
                        model.set_initial_conditions_from(soln, return_type="ics")
                    )
                    y0s.append(
                        concatenated_initial_conditions.evaluate(0, inputs=inputs)
                    )

            model.y0_list = y0s

        batched_inputs = [
            self._inputs_to_stacked_vect(
                model_inputs_list[i * batch_size : (i + 1) * batch_size],
                model.convert_to_format,
            )
            for i in range(len(inputs_list) // batch_size)
        ]

        set_up_time = timer.time()

        # (Re-)calculate consistent initial conditions
        self._set_initial_conditions(
            model,
            t_start_shifted,
            model_inputs_list,
            update_rhs=False,
            batched_inputs=batched_inputs,
        )

        # Check initial conditions don't violate events
        for y0, inputs in zip(model.y0_list, batched_inputs):
            self._check_events_with_initial_conditions(t_eval, model, y0, inputs)

        # Step
        pybamm.logger.verbose(f"Stepping for {t_start_shifted:.0f} < t < {t_end:.0f}")
        timer.reset()
        solutions = self._integrate(model, t_eval, model_inputs_list, batched_inputs)
        for i, s in enumerate(solutions):
            solutions[i].solve_time = timer.time()

            # Check if extrapolation occurred
            self.check_extrapolation(s, model.events)

            # Identify the event that caused termination and update the solution to
            # include the event time and state
            solutions[i], termination = self.get_termination_reason(s, model.events)

            # Assign setup time
            solutions[i].set_up_time = set_up_time

        # Report times
        pybamm.logger.verbose(f"Finish stepping {model.name} ({termination})")
        pybamm.logger.verbose(
            f"Set-up time: {solutions[0].set_up_time}, Step time: {solutions[0].solve_time} (of which integration time: {solutions[0].integration_time}), "
            f"Total time: {solutions[0].total_time}"
        )

        # Return solution
        if save is False:
            ret = solutions
        else:
            ret = [old_s + s for (old_s, s) in zip(old_solutions, solutions)]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

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
        elif solution.termination == "event":
            pybamm.logger.debug("Start post-processing events")
            if isinstance(solution.y_event, casadi.DM):
                solution_y_event = solution.y_event.full()
            else:
                solution_y_event = solution.y_event
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
                        solution_y_event,
                        inputs=solution.all_inputs[-1],
                    )
                termination_event = min(final_event_values, key=final_event_values.get)

                # Add the event to the solution object
                # Check that it's actually an event for this solution (might be from another input set)
                if final_event_values[termination_event] > 0.1:
                    solution.termination = (
                        f"event: {termination_event} from another input set"
                    )
                else:
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
                )
                event_sol.solve_time = 0.0
                event_sol.integration_time = 0.0
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

        if any(
            event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION
            for event in events
        ):
            last_state = solution.last_state
            t = last_state.all_ts[0][0]
            y = last_state.all_ys[0][:, 0]
            inputs = last_state.all_inputs[0]

            if isinstance(y, casadi.DM):
                y = y.full()
            for event in events:
                if event.event_type == pybamm.EventType.INTERPOLANT_EXTRAPOLATION:
                    if event.expression.evaluate(t, y, inputs=inputs) < self.extrap_tol:
                        extrap_events.append(event.name)

            if any(extrap_events):
                if self._on_extrapolation == "warn":
                    name = solution.all_models[-1].name
                    warnings.warn(
                        f"While solving {name} extrapolation occurred "
                        f"for {extrap_events}",
                        pybamm.SolverWarning,
                        stacklevel=2,
                    )
                    # Add the event dictionary to the solution object
                    solution.extrap_events = extrap_events
                elif self._on_extrapolation == "error":
                    raise pybamm.SolverError(
                        "Solver failed because the following "
                        f"interpolation bounds were exceeded: {extrap_events}. "
                        "You may need to provide additional interpolation points "
                        "outside these bounds."
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
            if name in inputs:
                inputs_in_model[name] = inputs[name]
            else:
                raise pybamm.SolverError(f"No value provided for input '{name}'")
        inputs = inputs_in_model

        ordered_inputs_names = list(inputs.keys())
        ordered_inputs_names.sort()
        ordered_inputs = {name: inputs[name] for name in ordered_inputs_names}

        return ordered_inputs

    @staticmethod
    def _inputs_to_stacked_vect(inputs_list: list[dict], convert_to_format: str):
        if len(inputs_list) == 0 or len(inputs_list[0]) == 0:
            return np.array([[]])
        if convert_to_format == "casadi":
            inputs = casadi.vertcat(
                *[x for inputs in inputs_list for x in inputs.values()]
            )
        else:
            arrays_to_stack = [
                np.array(x, dtype=float).reshape(-1, 1)
                for inputs in inputs_list
                for x in inputs.values()
            ]
            inputs = np.vstack(arrays_to_stack)
        return inputs

    @staticmethod
    def _input_dict_to_slices(input_dict: dict):
        input_slices = {}
        i = 0
        for key, value in input_dict.items():
            if isinstance(value, np.ndarray):
                inc = value.shape[0]
                input_slices[key] = slice(i, i + inc)
            else:
                inc = 1
                input_slices[key] = i
            i += inc
        return input_slices

    @staticmethod
    def _unzip_state_vector(model, y):
        nstates = (
            model.len_rhs + model.len_rhs_sens + model.len_alg + model.len_alg_sens
        )
        len_rhs = model.len_rhs + model.len_rhs_sens
        batch_size = model.batch_size

        if isinstance(y, casadi.DM):
            y_diff = casadi.vertcat(
                *[y[i * nstates : i * nstates + len_rhs] for i in range(batch_size)]
            )
            y_alg = casadi.vertcat(
                *[
                    y[i * nstates + len_rhs : (i + 1) * nstates]
                    for i in range(batch_size)
                ]
            )
        else:
            y_diff = np.vstack(
                [y[i * nstates : i * nstates + len_rhs] for i in range(batch_size)]
            )
            y_alg = np.vstack(
                [
                    y[i * nstates + len_rhs : (i + 1) * nstates]
                    for i in range(batch_size)
                ]
            )

        return y_diff, y_alg

    @staticmethod
    def _zip_state_vector(model, y_diff, y_alg):
        len_rhs = model.len_rhs + model.len_rhs_sens
        len_alg = model.len_alg + model.len_alg_sens
        batch_size = model.batch_size
        y_diff_list = [
            y_diff[i * len_rhs : (i + 1) * len_rhs, :] for i in range(batch_size)
        ]
        y_alg_list = [
            y_alg[i * len_alg : (i + 1) * len_alg, :] for i in range(batch_size)
        ]
        if isinstance(y_diff, casadi.DM):
            y = casadi.vertcat(
                *[val for pair in zip(y_diff_list, y_alg_list) for val in pair]
            )
        else:
            y = np.vstack(
                [val for pair in zip(y_diff_list, y_alg_list) for val in pair]
            )
        return y


def map_func_over_inputs_casadi(name, f, vars_for_processing, ninputs, nthreads):
    """
    This takes a casadi function f and returns a new casadi function that maps f over
    the provided number of inputs. Some functions (e.g. jacobian action) require an additional
    vector input v, which is why add_v is provided.

    Parameters
    ----------
    name: str
        name of the new function. This must end in the string "_action" for jacobian action functions,
        "_jac" for jacobian functions, or "_jacp" for jacp functions.
    f: casadi.Function
        function to map
    vars_for_processing: dict
        dictionary of variables for processing
    ninputs: int
        number of inputs to map over
    nthreads: int
        number of threads to use
    """
    if f is None:
        return None

    is_event = "event" in name
    add_v = name.endswith("_action")
    matrix_output = name.endswith("_jac") or name.endswith("_jacp")

    nstates = vars_for_processing["y_and_S"].shape[0]
    nparams = vars_for_processing["p_casadi_stacked"].shape[0]

    if nthreads > 1:
        parallelisation = "thread"
    else:
        parallelisation = "none"
    y_and_S_inputs_stacked = casadi.MX.sym("y_and_S_stacked", nstates * ninputs)
    p_casadi_inputs_stacked = casadi.MX.sym("p_stacked", nparams * ninputs)
    v_inputs_stacked = casadi.MX.sym("v_stacked", nstates * ninputs)
    t_stacked = casadi.MX.sym("t_stacked", ninputs)

    y_and_S_2d = y_and_S_inputs_stacked.reshape((nstates, ninputs))
    p_casadi_2d = p_casadi_inputs_stacked.reshape((nparams, ninputs))
    v_2d = v_inputs_stacked.reshape((nstates, ninputs))
    t_2d = t_stacked.reshape((1, ninputs))

    if add_v:
        inputs_2d = [t_2d, y_and_S_2d, p_casadi_2d, v_2d]
        inputs_stacked = [
            t_stacked,
            y_and_S_inputs_stacked,
            p_casadi_inputs_stacked,
            v_inputs_stacked,
        ]
    else:
        inputs_2d = [t_2d, y_and_S_2d, p_casadi_2d]
        inputs_stacked = [t_stacked, y_and_S_inputs_stacked, p_casadi_inputs_stacked]

    mapped_f = f.map(ninputs, parallelisation, nthreads)(*inputs_2d)
    if matrix_output:
        # for matrix output we need to stack the outputs in a block diagonal matrix
        splits = [i * nstates for i in range(ninputs + 1)]
        split = casadi.horzsplit(mapped_f, splits)
        stack = casadi.diagcat(*split)
    elif is_event:
        # Events need to return a scalar, so we combine the vector output
        # of the mapped function into a scalar output by calculating a smooth max of the vector output.
        stack = casadi.logsumexp(casadi.transpose(mapped_f), 1e-4)
    else:
        # for vector outputs we need to stack them vertically in a single column vector
        splits = [i for i in range(ninputs + 1)]
        split = casadi.horzsplit(mapped_f, splits)
        stack = casadi.vertcat(*split)
    return casadi.Function(name, inputs_stacked, [stack])


def process(
    symbol,
    name,
    vars_for_processing,
    inputs: list[dict],
    batch_size,
    nthreads,
    use_jacobian=None,
    return_jacp_stacked=None,
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
    inputs: list of dict
        list of input parameters to pass to the model when solving
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
    if batch_size > 1:
        inputs_batch = inputs[:batch_size]
    else:
        inputs_batch = [inputs[0]]
    is_event = "event" in name
    nbatches = len(inputs) // batch_size
    nthreads_per_batch = max(nthreads // nbatches, 1)

    def report(string):
        # don't log event conversion
        if "event" not in string:
            pybamm.logger.verbose(string)

    model = vars_for_processing["model"]

    if use_jacobian is None:
        use_jacobian = model.use_jacobian

    if model.convert_to_format == "jax":
        report(f"Converting {name} to jax")
        func = pybamm.EvaluatorJax(symbol, inputs_batch, is_event=is_event)
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
        # Process with pybamm functions, converting
        # to python evaluator
        if model.calculate_sensitivities:
            report(
                f"Calculating sensitivities for {name} with respect "
                f"to parameters {model.calculate_sensitivities}"
            )
            jacp_dict = {
                p: pybamm.EvaluatorPython(
                    symbol.diff(pybamm.InputParameter(p)), inputs_batch
                )
                for p in model.calculate_sensitivities
            }

            # jacp returns a matrix where each column is the sensitivity for
            # a different parameter
            def jacp(t, y, inputs):
                sens_list = [v(t, y, inputs) for v in jacp_dict.values()]
                return np.hstack(sens_list)
        else:
            jacp = None

        if use_jacobian:
            report(f"Calculating jacobian for {name}")
            jac = jacobian.jac(symbol, y)
            report(f"Converting jacobian for {name} to python")
            jac = pybamm.EvaluatorPython(jac, inputs_batch, is_matrix=True)
            # cannot do jacobian action efficiently for now
            jac_action = None
        else:
            jac = None
            jac_action = None

        report(f"Converting {name} to python")
        func = pybamm.EvaluatorPython(symbol, inputs_batch, is_event=is_event)

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
            jacp = casadi.Function(
                f"d{name}_dp",
                [t_casadi, y_casadi, p_casadi_stacked],
                [casadi.jacobian(casadi_expression, p_casadi_stacked)],
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

        if batch_size > 1:
            func = map_func_over_inputs_casadi(
                name, func, vars_for_processing, batch_size, nthreads_per_batch
            )
            jac = map_func_over_inputs_casadi(
                name + "_jac",
                jac,
                vars_for_processing,
                batch_size,
                nthreads_per_batch,
            )
            jacp = map_func_over_inputs_casadi(
                name + "_jacp",
                jacp,
                vars_for_processing,
                batch_size,
                nthreads_per_batch,
            )
            jac_action = map_func_over_inputs_casadi(
                name + "_jac_action",
                jac_action,
                vars_for_processing,
                batch_size,
                nthreads_per_batch,
            )

    return func, jac, jacp, jac_action
