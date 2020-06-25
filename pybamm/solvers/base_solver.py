#
# Base solver class
#
import casadi
import copy
import pybamm
import numbers
import numpy as np
import sys
import itertools


class BaseSolver(object):
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
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method=None,
        root_tol=1e-6,
        max_steps="deprecated",
    ):
        self._method = method
        self._rtol = rtol
        self._atol = atol
        self.root_tol = root_tol
        self.root_method = root_method
        if max_steps != "deprecated":
            raise ValueError(
                "max_steps has been deprecated, and should be set using the "
                "solver-specific extra-options dictionaries instead"
            )
        self.models_set_up = set()

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self.ode_solver = False
        self.algebraic_solver = False

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, value):
        self._rtol = value

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, value):
        self._atol = value

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

    @property
    def root_tol(self):
        return self._root_tol

    @root_tol.setter
    def root_tol(self, tol):
        self._root_tol = tol

    def copy(self):
        "Returns a copy of the solver"
        new_solver = copy.copy(self)
        # clear models_set_up
        new_solver.models_set_up = set()
        return new_solver

    def set_up(self, model, inputs=None):
        """Unpack model, perform checks, simplify and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        """

        # Check model.algebraic for ode solvers
        if self.ode_solver is True and len(model.algebraic) > 0:
            raise pybamm.SolverError(
                "Cannot use ODE solver '{}' to solve DAE model".format(self.name)
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
                    "model should be discretised before solving ({})".format(e)
                )

        inputs = inputs or {}

        # Set model timescale
        try:
            model.timescale_eval = model.timescale.evaluate()
        except KeyError as e:
            raise pybamm.SolverError(
                "The model timescale is a function of an input parameter "
                "(original error: {})".format(e)
            )

        if (
            isinstance(self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver))
        ) and model.convert_to_format != "casadi":
            pybamm.logger.warning(
                "Converting {} to CasADi for solving with CasADi solver".format(
                    model.name
                )
            )
            model.convert_to_format = "casadi"
        if (
            isinstance(self.root_method, pybamm.CasadiAlgebraicSolver)
            and model.convert_to_format != "casadi"
        ):
            pybamm.logger.warning(
                "Converting {} to CasADi for calculating ICs with CasADi".format(
                    model.name
                )
            )
            model.convert_to_format = "casadi"

        if model.convert_to_format != "casadi":
            simp = pybamm.Simplification()
            # Create Jacobian from concatenated rhs and algebraic
            y = pybamm.StateVector(slice(0, model.concatenated_initial_conditions.size))
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
        else:
            # Convert model attributes to casadi
            t_casadi = casadi.MX.sym("t")
            y_diff = casadi.MX.sym("y_diff", model.concatenated_rhs.size)
            y_alg = casadi.MX.sym("y_alg", model.concatenated_algebraic.size)
            y_casadi = casadi.vertcat(y_diff, y_alg)
            p_casadi = {}
            for name, value in inputs.items():
                if isinstance(value, numbers.Number):
                    p_casadi[name] = casadi.MX.sym(name)
                else:
                    p_casadi[name] = casadi.MX.sym(name, value.shape[0])
            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

        def process(func, name, use_jacobian=None):
            def report(string):
                # don't log event conversion
                if "event" not in string:
                    pybamm.logger.info(string)

            if use_jacobian is None:
                use_jacobian = model.use_jacobian
            if model.convert_to_format != "casadi":
                # Process with pybamm functions
                if model.use_simplify:
                    report(f"Simplifying {name}")
                    func = simp.simplify(func)
                if use_jacobian:
                    report(f"Calculating jacobian for {name}")
                    jac = jacobian.jac(func, y)
                    if model.use_simplify:
                        report(f"Simplifying jacobian for {name}")
                        jac = simp.simplify(jac)
                    if model.convert_to_format == "python":
                        report(f"Converting jacobian for {name} to python")
                        jac = pybamm.EvaluatorPython(jac)
                    jac = jac.evaluate
                else:
                    jac = None
                if model.convert_to_format == "python":
                    report(f"Converting {name} to python")
                    func = pybamm.EvaluatorPython(func)
                func = func.evaluate
            else:
                # Process with CasADi
                report(f"Converting {name} to CasADi")
                func = func.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
                if use_jacobian:
                    report(f"Calculating jacobian for {name} using CasADi")
                    jac_casadi = casadi.jacobian(func, y_casadi)
                    jac = casadi.Function(
                        name, [t_casadi, y_casadi, p_casadi_stacked], [jac_casadi]
                    )
                else:
                    jac = None
                func = casadi.Function(
                    name, [t_casadi, y_casadi, p_casadi_stacked], [func]
                )
            if name == "residuals":
                func_call = Residuals(func, name, model)
            else:
                func_call = SolverCallable(func, name, model)
            if jac is not None:
                jac_call = SolverCallable(jac, name + "_jac", model)
            else:
                jac_call = None
            return func, func_call, jac_call

        # Check for heaviside functions in rhs and algebraic and add discontinuity
        # events if these exist.
        # Note: only checks for the case of t < X, t <= X, X < t, or X <= t, but also
        # accounts for the fact that t might be dimensional
        # Only do this for DAE models as ODE models can deal with discontinuities fine
        if len(model.algebraic) > 0:
            for symbol in itertools.chain(
                model.concatenated_rhs.pre_order(),
                model.concatenated_algebraic.pre_order(),
            ):
                if isinstance(symbol, pybamm.Heaviside):
                    found_t = False
                    # Dimensionless
                    if symbol.right.id == pybamm.t.id:
                        expr = symbol.left
                        found_t = True
                    elif symbol.left.id == pybamm.t.id:
                        expr = symbol.right
                        found_t = True
                    # Dimensional
                    elif symbol.right.id == (pybamm.t * model.timescale).id:
                        expr = symbol.left.new_copy() / symbol.right.right.new_copy()
                        found_t = True
                    elif symbol.left.id == (pybamm.t * model.timescale).id:
                        expr = symbol.right.new_copy() / symbol.left.right.new_copy()
                        found_t = True

                    # Update the events if the heaviside function depended on t
                    if found_t:
                        model.events.append(
                            pybamm.Event(
                                str(symbol),
                                expr.new_copy(),
                                pybamm.EventType.DISCONTINUITY,
                            )
                        )

        # Process initial conditions
        initial_conditions = process(
            model.concatenated_initial_conditions,
            "initial_conditions",
            use_jacobian=False,
        )[0]
        init_eval = InitialConditions(initial_conditions, model)

        # Process rhs, algebraic and event expressions
        rhs, rhs_eval, jac_rhs = process(model.concatenated_rhs, "RHS")
        algebraic, algebraic_eval, jac_algebraic = process(
            model.concatenated_algebraic, "algebraic"
        )
        terminate_events_eval = [
            process(event.expression, "event", use_jacobian=False)[1]
            for event in model.events
            if event.event_type == pybamm.EventType.TERMINATION
        ]

        # discontinuity events are evaluated before the solver is called, so don't need
        # to process them
        discontinuity_events_eval = [
            event
            for event in model.events
            if event.event_type == pybamm.EventType.DISCONTINUITY
        ]

        # Add the solver attributes
        model.init_eval = init_eval
        model.rhs_eval = rhs_eval
        model.algebraic_eval = algebraic_eval
        model.jac_algebraic_eval = jac_algebraic
        model.terminate_events_eval = terminate_events_eval
        model.discontinuity_events_eval = discontinuity_events_eval

        # Calculate initial conditions
        model.y0 = init_eval(inputs)

        # Save CasADi functions for the CasADi solver
        # Note: when we pass to casadi the ode part of the problem must be in explicit
        # form so we pre-multiply by the inverse of the mass matrix
        if isinstance(self.root_method, pybamm.CasadiAlgebraicSolver) or isinstance(
            self, (pybamm.CasadiSolver, pybamm.CasadiAlgebraicSolver)
        ):
            # can use DAE solver to solve model with algebraic equations only
            if len(model.rhs) > 0:
                mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
                explicit_rhs = mass_matrix_inv @ rhs(
                    t_casadi, y_casadi, p_casadi_stacked
                )
                model.casadi_rhs = casadi.Function(
                    "rhs", [t_casadi, y_casadi, p_casadi_stacked], [explicit_rhs]
                )
            model.casadi_algebraic = algebraic
        if len(model.rhs) == 0:
            # No rhs equations: residuals is algebraic only
            model.residuals_eval = Residuals(algebraic, "residuals", model)
            model.jacobian_eval = jac_algebraic
        elif len(model.algebraic) == 0:
            # No algebraic equations: residuals is rhs only
            model.residuals_eval = Residuals(rhs, "residuals", model)
            model.jacobian_eval = jac_rhs
        # Calculate consistent initial conditions for the algebraic equations
        else:
            all_states = pybamm.NumpyConcatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )
            # Process again, uses caching so should be quick
            residuals_eval, jacobian_eval = process(all_states, "residuals")[1:]
            model.residuals_eval = residuals_eval
            model.jacobian_eval = jacobian_eval

        pybamm.logger.info("Finish solver set-up")

    def _set_initial_conditions(self, model, inputs, update_rhs):
        """
        Set initial conditions for the model. This is skipped if the solver is an
        algebraic solver (since this would make the algebraic solver redundant), and if
        the model doesn't have any algebraic equations (since there are no initial
        conditions to be calculated in this case).

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        inputs : dict
            Any input parameters to pass to the model when solving
        update_rhs : bool
            Whether to update the rhs. True for 'solve', False for 'step'.

        """
        if self.algebraic_solver is True:
            # Don't update model.y0
            return None
        elif len(model.algebraic) == 0:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                model.y0 = model.init_eval(inputs)
            else:
                # Don't update model.y0
                return None
        else:
            if update_rhs is True:
                # Recalculate initial conditions for the rhs equations
                y0_from_inputs = model.init_eval(inputs)
                # Reuse old solution for algebraic equations
                y0_from_model = model.y0
                len_rhs = model.concatenated_rhs.size
                # update model.y0, which is used for initialising the algebraic solver
                if len_rhs == 0:
                    model.y0 = y0_from_model
                else:
                    model.y0 = casadi.vertcat(
                        y0_from_inputs[:len_rhs], y0_from_model[len_rhs:]
                    )
            model.y0 = self.calculate_consistent_state(model, 0, inputs)

    def calculate_consistent_state(self, model, time=0, inputs=None):
        """
        Calculate consistent state for the algebraic equations through
        root-finding. model.y0 is used as the initial guess for rootfinding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.
        time : float
            The time at which to calculate the states
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations)
        """
        pybamm.logger.info("Start calculating consistent states")
        try:
            root_sol = self.root_method._integrate(model, [time], inputs)
        except pybamm.SolverError as e:
            raise pybamm.SolverError(
                "Could not find consistent initial conditions: {}".format(e.args[0])
            )
        return root_sol.y.flatten()

    def solve(self, model, t_eval=None, external_variables=None, inputs=None):
        """
        Execute the solver setup and calculate the solution of the model at
        specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times (in seconds) at which to compute the solution
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}` and
            `model.variables = {}`)

        """
        pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
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

        # Make sure t_eval is monotonic
        if (np.diff(t_eval) < 0).any():
            raise pybamm.SolverError("t_eval must increase monotonically")

        # Set up external variables and inputs
        ext_and_inputs = self._set_up_ext_and_inputs(model, external_variables, inputs)

        # Set up
        timer = pybamm.Timer()

        # Set up (if not done already)
        if model not in self.models_set_up:
            self.set_up(model, ext_and_inputs)
            set_up_time = timer.time()
            self.models_set_up.add(model)
        else:
            set_up_time = 0

        # (Re-)calculate consistent initial conditions
        self._set_initial_conditions(model, ext_and_inputs, update_rhs=True)

        # Non-dimensionalise time
        t_eval_dimensionless = t_eval / model.timescale_eval

        # Calculate discontinuities
        discontinuities = [
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
        discontinuities = [v for v in discontinuities if v < t_eval_dimensionless[-1]]

        if len(discontinuities) > 0:
            pybamm.logger.info(
                "Discontinuity events found at t = {}".format(discontinuities)
            )
        else:
            pybamm.logger.info("No discontinuity events found")

        # insert time points around discontinuities in t_eval
        # keep track of sub sections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        eps = sys.float_info.epsilon
        for dtime in discontinuities:
            dindex = np.searchsorted(t_eval_dimensionless, dtime, side="left")
            end_indices.append(dindex + 1)
            start_indices.append(dindex + 1)
            if dtime - eps < t_eval_dimensionless[dindex] < dtime + eps:
                t_eval_dimensionless[dindex] += eps
                t_eval_dimensionless = np.insert(
                    t_eval_dimensionless, dindex, dtime - eps
                )
            else:
                t_eval_dimensionless = np.insert(
                    t_eval_dimensionless, dindex, [dtime - eps, dtime + eps]
                )
        end_indices.append(len(t_eval_dimensionless))

        # integrate separately over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a dae)
        old_y0 = model.y0
        solution = None
        for start_index, end_index in zip(start_indices, end_indices):
            pybamm.logger.info(
                "Calling solver for {} < t < {}".format(
                    t_eval_dimensionless[start_index] * model.timescale_eval,
                    t_eval_dimensionless[end_index - 1] * model.timescale_eval,
                )
            )
            timer.reset()
            new_solution = self._integrate(
                model, t_eval_dimensionless[start_index:end_index], ext_and_inputs
            )
            new_solution.solve_time = timer.time()
            if solution is None:
                solution = new_solution
            else:
                solution.append(new_solution, start_index=0)

            if solution.termination != "final time":
                break

            if end_index != len(t_eval_dimensionless):
                # setup for next integration subsection
                last_state = solution.y[:, -1]
                # update y0 (for DAE solvers, this updates the initial guess for the
                # rootfinder)
                model.y0 = last_state
                if len(model.algebraic) > 0:
                    model.y0 = self.calculate_consistent_state(
                        model, t_eval_dimensionless[end_index], ext_and_inputs
                    )

        # restore old y0
        model.y0 = old_y0

        # Assign times
        solution.set_up_time = set_up_time
        solution.solve_time = timer.time()

        # Add model and inputs to solution
        solution.model = model
        solution.inputs = ext_and_inputs

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, model.events)

        pybamm.logger.info("Finish solving {} ({})".format(model.name, termination))
        pybamm.logger.info(
            "Set-up time: {}, Solve time: {}, Total time: {}".format(
                timer.format(solution.set_up_time),
                timer.format(solution.solve_time),
                timer.format(solution.total_time),
            )
        )

        # Raise error if solution only contains one timestep (except for algebraic
        # solvers, where we may only expect one time in the solution)
        if self.algebraic_solver is False and len(solution.t) == 1:
            raise pybamm.SolverError(
                "Solution time vector has length 1. "
                "Check whether simulation terminated too early."
            )

        return solution

    def step(
        self,
        old_solution,
        model,
        dt,
        npts=2,
        external_variables=None,
        inputs=None,
        save=True,
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
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. default is 2 (returns the solution at t0 and t0 + dt).
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool
            Turn on to store the solution of all previous timesteps

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

        """

        if old_solution is not None and not (
            old_solution.termination == "final time"
            or "[experiment]" in old_solution.termination
        ):
            # Return same solution as an event has already been triggered
            # With hack to allow stepping past experiment current / voltage cut-off
            return old_solution

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            if not isinstance(self, pybamm.DummySolver):
                raise pybamm.ModelError(
                    "Cannot step empty model, use `pybamm.DummySolver` instead"
                )

        # Set timer
        timer = pybamm.Timer()

        # Set up external variables and inputs
        external_variables = external_variables or {}
        inputs = inputs or {}
        ext_and_inputs = {**external_variables, **inputs}

        # Run set up on first step
        if old_solution is None:
            pybamm.logger.info(
                "Start stepping {} with {}".format(model.name, self.name)
            )
            self.set_up(model, ext_and_inputs)
            t = 0.0
            set_up_time = timer.time()
        else:
            # initialize with old solution
            t = old_solution.t[-1]
            model.y0 = old_solution.y[:, -1]
            set_up_time = 0

        # (Re-)calculate consistent initial conditions
        self._set_initial_conditions(model, ext_and_inputs, update_rhs=False)

        # Non-dimensionalise dt
        dt_dimensionless = dt / model.timescale_eval

        # Step
        t_eval = np.linspace(t, t + dt_dimensionless, npts)
        pybamm.logger.info("Calling solver")
        timer.reset()
        solution = self._integrate(model, t_eval, ext_and_inputs)

        # Assign times
        solution.set_up_time = set_up_time
        solution.solve_time = timer.time()

        # Add model and inputs to solution
        solution.model = model
        solution.inputs = ext_and_inputs

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, model.events)

        pybamm.logger.debug("Finish stepping {} ({})".format(model.name, termination))
        if set_up_time:
            pybamm.logger.debug(
                "Set-up time: {}, Step time: {}, Total time: {}".format(
                    timer.format(solution.set_up_time),
                    timer.format(solution.solve_time),
                    timer.format(solution.total_time),
                )
            )
        else:
            pybamm.logger.debug(
                "Step time: {}".format(timer.format(solution.solve_time))
            )
        if save is False or old_solution is None:
            return solution
        else:
            return old_solution + solution

    def get_termination_reason(self, solution, events):
        """
        Identify the cause for termination. In particular, if the solver terminated
        due to an event, (try to) pinpoint which event was responsible.
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
        if solution.termination == "final time":
            return "the solver successfully reached the end of the integration interval"
        elif solution.termination == "event":
            # Get final event value
            final_event_values = {}

            for event in events:
                if event.event_type == pybamm.EventType.TERMINATION:
                    final_event_values[event.name] = abs(
                        event.expression.evaluate(
                            solution.t_event,
                            solution.y_event,
                            inputs={k: v[:, -1] for k, v in solution.inputs.items()},
                        )
                    )
            termination_event = min(final_event_values, key=final_event_values.get)
            # Add the event to the solution object
            solution.termination = "event: {}".format(termination_event)
            return "the termination event '{}' occurred".format(termination_event)

    def _set_up_ext_and_inputs(self, model, external_variables, inputs):
        "Set up external variables and input parameters"
        inputs = inputs or {}

        # Go through all input parameters that can be found in the model
        # If any of them are *not* provided by "inputs", a symbolic input parameter is
        # created, with appropriate size
        for input_param in model.input_parameters:
            name = input_param.name
            if name not in inputs:
                # Only allow symbolic inputs for CasadiAlgebraicSolver
                if not isinstance(self, pybamm.CasadiAlgebraicSolver):
                    raise pybamm.SolverError(
                        "Only CasadiAlgebraicSolver can have symbolic inputs"
                    )
                inputs[name] = casadi.MX.sym(name, input_param._expected_size)

        external_variables = external_variables or {}
        ext_and_inputs = {**external_variables, **inputs}
        return ext_and_inputs


class SolverCallable:
    "A class that will be called by the solver when integrating"

    def __init__(self, function, name, model):
        self._function = function
        if isinstance(function, casadi.Function):
            self.form = "casadi"
        else:
            self.form = "python"
        self.name = name
        self.model = model
        self.timescale = self.model.timescale_eval

    def __call__(self, t, y, inputs):
        y = y[:, np.newaxis]
        if self.name in ["RHS", "algebraic", "residuals", "event"]:
            pybamm.logger.debug(
                "Evaluating {} for {} at t={}".format(
                    self.name, self.model.name, t * self.timescale
                )
            )
            return self.function(t, y, inputs).flatten()
        else:
            return self.function(t, y, inputs)

    def function(self, t, y, inputs):
        if self.form == "casadi":
            states_eval = self._function(t, y, inputs)
            if self.name in ["RHS", "algebraic", "residuals", "event"]:
                return states_eval.full()
            else:
                # keep jacobians sparse
                return states_eval
        else:
            return self._function(t, y, inputs=inputs, known_evals={})[0]


class Residuals(SolverCallable):
    "Returns information about residuals at time t and state y"

    def __init__(self, function, name, model):
        super().__init__(function, name, model)
        if model.mass_matrix is not None:
            self.mass_matrix = model.mass_matrix.entries

    def __call__(self, t, y, ydot, inputs):
        states_eval = super().__call__(t, y, inputs)
        return states_eval - self.mass_matrix @ ydot


class InitialConditions(SolverCallable):
    "Returns initial conditions given inputs"

    def __init__(self, function, model):
        super().__init__(function, "initial conditions", model)
        self.y_dummy = np.zeros(model.concatenated_initial_conditions.shape)

    def __call__(self, inputs):
        if self.form == "casadi":
            if isinstance(inputs, dict):
                inputs = casadi.vertcat(*[x for x in inputs.values()])
            return self._function(0, self.y_dummy, inputs)
        else:
            return self._function(0, self.y_dummy, inputs=inputs).flatten()
