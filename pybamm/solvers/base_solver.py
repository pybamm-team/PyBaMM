#
# Base solver class
#
import casadi
import pybamm
import numbers
import numpy as np
from scipy import optimize
from scipy.sparse import issparse
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
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method="lm",
        root_tol=1e-6,
        max_steps=1000,
    ):
        self._method = method
        self._rtol = rtol
        self._atol = atol
        self.root_method = root_method
        self.root_tol = root_tol
        self.max_steps = max_steps

        self.models_set_up = set()
        self.model_step_times = {}

        # Defaults, can be overwritten by specific solver
        self.name = "Base solver"
        self.ode_solver = False

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
        self._root_method = method

    @property
    def root_tol(self):
        return self._root_tol

    @root_tol.setter
    def root_tol(self, tol):
        self._root_tol = tol

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps

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
        inputs = inputs or {}
        y0 = model.concatenated_initial_conditions

        # Check model.algebraic for ode solvers
        if self.ode_solver is True and len(model.algebraic) > 0:
            raise pybamm.SolverError(
                "Cannot use ODE solver '{}' to solve DAE model".format(self.name)
            )

        if (
            isinstance(self, pybamm.CasadiSolver)
            and model.convert_to_format != "casadi"
        ):
            pybamm.logger.warning(
                f"Converting {model.name} to CasADi for solving with CasADi solver"
            )
            model.convert_to_format = "casadi"

        if model.convert_to_format != "casadi":
            simp = pybamm.Simplification()
            # Create Jacobian from concatenated rhs and algebraic
            y = pybamm.StateVector(
                slice(0, np.size(model.concatenated_initial_conditions))
            )
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
        else:
            # Convert model attributes to casadi
            t_casadi = casadi.MX.sym("t")
            y_diff = casadi.MX.sym(
                "y_diff", len(model.concatenated_rhs.evaluate(0, y0, inputs))
            )
            y_alg = casadi.MX.sym(
                "y_alg", len(model.concatenated_algebraic.evaluate(0, y0, inputs))
            )
            y_casadi = casadi.vertcat(y_diff, y_alg)
            u_casadi = {}
            for name, value in inputs.items():
                if isinstance(value, numbers.Number):
                    u_casadi[name] = casadi.MX.sym(name)
                else:
                    u_casadi[name] = casadi.MX.sym(name, value.shape[0])
            u_casadi_stacked = casadi.vertcat(*[u for u in u_casadi.values()])

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
                func = func.to_casadi(t_casadi, y_casadi, u_casadi)
                if use_jacobian:
                    report(f"Calculating jacobian for {name} using CasADi")
                    jac_casadi = casadi.jacobian(func, y_casadi)
                    jac = casadi.Function(
                        name, [t_casadi, y_casadi, u_casadi_stacked], [jac_casadi]
                    )
                else:
                    jac = None
                func = casadi.Function(
                    name, [t_casadi, y_casadi, u_casadi_stacked], [func]
                )
            if name == "residuals":
                func_call = Residuals(func, name, model)
            else:
                func_call = SolverCallable(func, name, model)
            func_call.set_inputs(inputs)
            if jac is not None:
                jac_call = SolverCallable(jac, name + "_jac", model)
                jac_call.set_inputs(inputs)
            else:
                jac_call = None
            return func, func_call, jac_call

        # Check for heaviside functions in rhs and algebraic and add discontinuity
        # events if these exist.
        # Note: only checks for the case of t < X, t <= X, X < t, or X <= t
        for symbol in itertools.chain(model.concatenated_rhs.pre_order(),
                                      model.concatenated_algebraic.pre_order()):
            if isinstance(symbol, pybamm.Heaviside):
                if symbol.right.id == pybamm.t.id:
                    expr = symbol.left
                elif symbol.left.id == pybamm.t.id:
                    expr = symbol.right

                model.events.append(pybamm.Event(str(symbol), expr.new_copy(),
                                                 pybamm.EventType.DISCONTINUITY))

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
            event for event in model.events
            if event.event_type == pybamm.EventType.DISCONTINUITY
        ]

        # Add the solver attributes
        model.rhs_eval = rhs_eval
        model.algebraic_eval = algebraic_eval
        model.jac_algebraic_eval = jac_algebraic
        model.terminate_events_eval = terminate_events_eval
        model.discontinuity_events_eval = discontinuity_events_eval

        # Calculate consistent initial conditions for the algebraic equations
        if len(model.algebraic) > 0:
            all_states = pybamm.NumpyConcatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )
            # Process again, uses caching so should be quick
            residuals, residuals_eval, jacobian_eval = process(all_states, "residuals")
            model.residuals_eval = residuals_eval
            model.jacobian_eval = jacobian_eval
            y0_guess = model.concatenated_initial_conditions.flatten()
            model.y0 = self.calculate_consistent_state(model, 0, y0_guess)
        else:
            # can use DAE solver to solve ODE model
            model.residuals_eval = Residuals(rhs, "residuals", model)
            model.jacobian_eval = jac_rhs
            model.y0 = model.concatenated_initial_conditions[:, 0]

        # Save CasADi functions for the CasADi solver
        # Note: when we pass to casadi the ode part of the problem must be in explicit
        # form so we pre-multiply by the inverse of the mass matrix
        if model.convert_to_format == "casadi" and isinstance(
            self, pybamm.CasadiSolver
        ):
            mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
            explicit_rhs = mass_matrix_inv @ rhs(t_casadi, y_casadi, u_casadi_stacked)
            model.casadi_rhs = casadi.Function(
                "rhs", [t_casadi, y_casadi, u_casadi_stacked], [explicit_rhs]
            )
            model.casadi_algebraic = algebraic

        pybamm.logger.info("Finish solver set-up")

    def set_inputs(self, model, ext_and_inputs):
        """
        Set values that are controlled externally, such as external variables and input
        parameters

        Parameters
        ----------
        ext_and_inputs : dict
            Any external variables or input parameters to pass to the model when solving

        """
        model.rhs_eval.set_inputs(ext_and_inputs)
        model.algebraic_eval.set_inputs(ext_and_inputs)
        model.residuals_eval.set_inputs(ext_and_inputs)
        for evnt in model.terminate_events_eval:
            evnt.set_inputs(ext_and_inputs)
        if model.jacobian_eval:
            model.jacobian_eval.set_inputs(ext_and_inputs)

    def calculate_consistent_state(self, model, time=0, y0_guess=None):
        """
        Calculate consistent state for the algebraic equations through
        root-finding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model for which to calculate initial conditions.

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations)
        """
        pybamm.logger.info("Start calculating consistent initial conditions")
        rhs = model.rhs_eval
        algebraic = model.algebraic_eval
        jac = model.jac_algebraic_eval
        if y0_guess is None:
            y0_guess = model.concatenated_initial_conditions.flatten()

        # Split y0_guess into differential and algebraic
        len_rhs = rhs(0, y0_guess).shape[0]
        y0_diff, y0_alg_guess = np.split(y0_guess, [len_rhs])

        def root_fun(y0_alg):
            "Evaluates algebraic using y0_diff (fixed) and y0_alg (changed by algo)"
            y0 = np.concatenate([y0_diff, y0_alg])
            out = algebraic(time, y0)
            pybamm.logger.debug(
                "Evaluating algebraic equations at t=0, L2-norm is {}".format(
                    np.linalg.norm(out)
                )
            )
            return out

        if jac:
            if issparse(jac(0, y0_guess)):

                def jac_fn(y0_alg):
                    """
                    Evaluates jacobian using y0_diff (fixed) and y0_alg (varying)
                    """
                    y0 = np.concatenate([y0_diff, y0_alg])
                    return jac(0, y0)[:, len_rhs:].toarray()

            else:

                def jac_fn(y0_alg):
                    """
                    Evaluates jacobian using y0_diff (fixed) and y0_alg (varying)
                    """
                    y0 = np.concatenate([y0_diff, y0_alg])
                    return jac(0, y0)[:, len_rhs:]

        else:
            jac_fn = None
        # Find the values of y0_alg that are roots of the algebraic equations
        sol = optimize.root(
            root_fun,
            y0_alg_guess,
            jac=jac_fn,
            method=self.root_method,
            tol=self.root_tol,
        )
        # Return full set of consistent initial conditions (y0_diff unchanged)
        y0_consistent = np.concatenate([y0_diff, sol.x])

        if sol.success and np.all(sol.fun < self.root_tol * len(sol.x)):
            pybamm.logger.info("Finish calculating consistent initial conditions")
            return y0_consistent
        elif not sol.success:
            raise pybamm.SolverError(
                "Could not find consistent initial conditions: {}".format(sol.message)
            )
        else:
            raise pybamm.SolverError(
                """
                Could not find consistent initial conditions: solver terminated
                successfully, but maximum solution error ({}) above tolerance ({})
                """.format(
                    np.max(sol.fun), self.root_tol * len(sol.x)
                )
            )

    def solve(self, model, t_eval, external_variables=None, inputs=None):
        """
        Execute the solver setup and calculate the solution of the model at
        specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}`)

        """
        pybamm.logger.info("Start solving {} with {}".format(model.name, self.name))

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            raise pybamm.ModelError("Cannot solve empty model")

        # Make sure t_eval is monotonic
        if (np.diff(t_eval) < 0).any():
            raise pybamm.SolverError("t_eval must increase monotonically")

        # Set up
        timer = pybamm.Timer()

        # Set up external variables and inputs
        external_variables = external_variables or {}
        inputs = inputs or {}
        ext_and_inputs = {**external_variables, **inputs}

        # Set up (if not done already)
        if model not in self.models_set_up:
            self.set_up(model, ext_and_inputs)
            set_up_time = timer.time()
            self.models_set_up.add(model)
        else:
            set_up_time = 0
        # Solve
        # Set inputs and external
        self.set_inputs(model, ext_and_inputs)

        # Calculate discontinuities
        discontinuities = [
            event.expression.evaluate(u=inputs)
            for event in model.discontinuity_events_eval
        ]

        # make sure they are increasing in time
        discontinuities = sorted(discontinuities)
        pybamm.logger.info(
            'Discontinuity events found at t = {}'.format(discontinuities)
        )
        # remove any identical discontinuities
        discontinuities = [
            v for i, v in enumerate(discontinuities)
            if i == len(discontinuities) - 1
            or discontinuities[i] < discontinuities[i + 1]
        ]

        # insert time points around discontinuities in t_eval
        # keep track of sub sections to integrate by storing start and end indices
        start_indices = [0]
        end_indices = []
        for dtime in discontinuities:
            dindex = np.searchsorted(t_eval, dtime, side='left')
            end_indices.append(dindex + 1)
            start_indices.append(dindex + 1)
            if t_eval[dindex] == dtime:
                t_eval[dindex] += sys.float_info.epsilon
                t_eval = np.insert(t_eval, dindex, dtime - sys.float_info.epsilon)
            else:
                t_eval = np.insert(t_eval, dindex,
                                   [dtime - sys.float_info.epsilon,
                                    dtime + sys.float_info.epsilon])
        end_indices.append(len(t_eval))

        # integrate separatly over each time segment and accumulate into the solution
        # object, restarting the solver at each discontinuity (and recalculating a
        # consistent state afterwards if a dae)
        old_y0 = model.y0
        solution = None
        for start_index, end_index in zip(start_indices, end_indices):
            pybamm.logger.info("Calling solver for {} < t < {}"
                               .format(t_eval[start_index], t_eval[end_index - 1]))
            timer.reset()
            if solution is None:
                solution = self._integrate(
                    model, t_eval[start_index:end_index], ext_and_inputs)
                solution.solve_time = timer.time()
            else:
                new_solution = self._integrate(
                    model, t_eval[start_index:end_index], ext_and_inputs)
                new_solution.solve_time = timer.time()
                solution.append(new_solution, start_index=0)

            if solution.termination != "final time":
                break

            if end_index != len(t_eval):
                # setup for next integration subsection
                y0_guess = solution.y[:, -1]
                if model.algebraic:
                    model.y0 = self.calculate_consistent_state(
                        model, t_eval[end_index], y0_guess)
                else:
                    model.y0 = y0_guess

                last_state = solution.y[:, -1]
                if len(model.algebraic) > 0:
                    model.y0 = self.calculate_consistent_state(
                        model, t_eval[end_index], last_state)
                else:
                    model.y0 = last_state

        # restore old y0
        model.y0 = old_y0

        # Assign times
        solution.set_up_time = set_up_time

        # Add model and inputs to solution
        solution.model = model
        solution.inputs = inputs

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
        return solution

    def step(
        self, old_solution, model, dt, npts=2, external_variables=None, inputs=None
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
            The timestep over which to step the solution
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. default is 2 (returns the solution at t0 and t0 + dt).
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time
        inputs : dict, optional
            Any input parameters to pass to the model when solving


        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}`)

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
            raise pybamm.ModelError("Cannot step empty model")

        # Set timer
        timer = pybamm.Timer()

        # Set up external variables and inputs
        external_variables = external_variables or {}
        inputs = inputs or {}
        ext_and_inputs = {**external_variables, **inputs}

        # Run set up on first step
        if model not in self.model_step_times:
            pybamm.logger.info(
                "Start stepping {} with {}".format(model.name, self.name)
            )
            self.set_up(model, ext_and_inputs)
            self.model_step_times[model] = 0.0
            set_up_time = timer.time()
        else:
            set_up_time = 0

        # Step
        t = self.model_step_times[model]
        t_eval = np.linspace(t, t + dt, npts)
        # Set inputs and external
        self.set_inputs(model, ext_and_inputs)

        pybamm.logger.info("Calling solver")
        timer.reset()
        solution = self._integrate(model, t_eval, ext_and_inputs)

        # Assign times
        solution.set_up_time = set_up_time
        solution.solve_time = timer.time()

        # Add model and inputs to solution
        solution.model = model
        solution.inputs = inputs

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, model.events)

        # Set self.t and self.y0 to their values at the final step
        self.model_step_times[model] = solution.t[-1]
        model.y0 = solution.y[:, -1]

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
        if old_solution is None:
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
                            {k: v[-1] for k, v in solution.inputs.items()},
                        )
                    )
            termination_event = min(final_event_values, key=final_event_values.get)
            # Add the event to the solution object
            solution.termination = "event: {}".format(termination_event)
            return "the termination event '{}' occurred".format(termination_event)


class SolverCallable:
    "A class that will be called by the solver when integrating"

    def __init__(self, function, name, model):
        self._function = function
        if isinstance(function, casadi.Function):
            self.form = "casadi"
            self.inputs = casadi.DM()
        else:
            self.form = "python"
            self.inputs = {}
        self.name = name
        self.model = model

    def set_inputs(self, inputs):
        "Set inputs"
        if self.form == "python":
            self.inputs = inputs
        elif self.form == "casadi":
            self.inputs = casadi.vertcat(*[x for x in inputs.values()])

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        if self.name in ["RHS", "algebraic", "residuals", "event"]:
            return self.function(t, y).flatten()
        else:
            return self.function(t, y)

    def function(self, t, y):
        if self.form == "casadi":
            if self.name in ["RHS", "algebraic", "residuals", "event"]:
                return self._function(t, y, self.inputs).full()
            else:
                # keep jacobians sparse
                return self._function(t, y, self.inputs)
        else:
            return self._function(t, y, self.inputs, known_evals={})[0]


class Residuals(SolverCallable):
    "Returns information about residuals at time t and state y"

    def __init__(self, function, name, model):
        super().__init__(function, name, model)
        self.mass_matrix = model.mass_matrix.entries

    def __call__(self, t, y, ydot):
        pybamm.logger.debug(
            "Evaluating residuals for {} at t={}".format(self.model.name, t)
        )
        states_eval = super().__call__(t, y)
        return states_eval - self.mass_matrix @ ydot
