#
# Base solver class
#
import casadi
import pybamm
import numpy as np
from scipy import optimize
from scipy.sparse import issparse


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

        self.name = "Base solver"

        self.y_pad = None
        self.y_ext = None

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
        y0 = add_external(y0, self.y_pad, self.y_ext)

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
            if self.y_pad is not None:
                y_ext = casadi.MX.sym("y_ext", len(self.y_pad))
                y_casadi_w_ext = casadi.vertcat(y_casadi, y_ext)
            else:
                y_casadi_w_ext = y_casadi
            u_casadi = {name: casadi.MX.sym(name) for name in inputs.keys()}
            u_casadi_stacked = casadi.vertcat(*[u for u in u_casadi.values()])

        def process(func, name, use_jacobian=None):
            if use_jacobian is None:
                use_jacobian = model.use_jacobian
            if model.convert_to_format != "casadi":
                # Process with pybamm functions
                if model.use_simplify:
                    pybamm.logger.info(f"Simplifying {name}")
                    func = simp.simplify(func)
                if use_jacobian:
                    pybamm.logger.info(f"Calculating jacobian for {name}")
                    jac = jacobian.jac(func, y)
                    if model.use_simplify:
                        pybamm.logger.info(f"Simplifying jacobian for {name}")
                        jac = simp.simplify(jac)
                    if model.convert_to_format == "python":
                        pybamm.logger.info(f"Converting jacobian for {name} to python")
                        jac = pybamm.EvaluatorPython(jac)
                    jac = jac.evaluate
                else:
                    jac = None
                if model.convert_to_format == "python":
                    pybamm.logger.info(f"Converting {name} to python")
                    func = pybamm.EvaluatorPython(func)
                func = func.evaluate
            else:
                # Process with CasADi
                pybamm.logger.info(f"Converting {name} to CasADi")
                func = func.to_casadi(t_casadi, y_casadi_w_ext, u_casadi)
                pybamm.logger.info(f"Converting jacobian for {name}")
                if use_jacobian:
                    jac_casadi = casadi.jacobian(func, y_casadi)
                    jac = casadi.Function(
                        name, [t_casadi, y_casadi_w_ext, u_casadi_stacked], [jac_casadi]
                    )
                else:
                    jac = None
                func = casadi.Function(
                    name, [t_casadi, y_casadi_w_ext, u_casadi_stacked], [func]
                )
            func_call = SolverCallable(func, name, model)
            func_call.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
            if jac is not None:
                jac_call = SolverCallable(jac, name + "_jac", model)
                jac_call.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
            else:
                jac_call = None
            return func, func_call, jac_call

        # Process rhs, algebraic and event expressions
        rhs, rhs_eval, jac_rhs = process(model.concatenated_rhs, "RHS")
        algebraic, algebraic_eval, jac_algebraic = process(
            model.concatenated_algebraic, "algebraic"
        )
        events_eval = [
            process(event, "event", use_jacobian=False)[1]
            for event in model.events.values()
        ]

        # Add the solver attributes
        model.rhs_eval = rhs_eval
        model.algebraic_eval = algebraic_eval
        model.jac_algebraic_eval = jac_algebraic
        model.events_eval = events_eval

        # Calculate consistent initial conditions for the algebraic equations
        if len(model.algebraic) > 0:
            all_states = pybamm.NumpyConcatenation(
                model.concatenated_rhs, model.concatenated_algebraic
            )
            # Process again, uses caching so should be quick
            residuals, residuals_eval, jacobian_eval = process(all_states, "residuals",)
            model.residuals_eval = residuals_eval
            model.jacobian_eval = jacobian_eval
            model.y0 = self.calculate_consistent_initial_conditions(model)
        else:
            # can use DAE solver to solve ODE model
            model.y0 = model.concatenated_initial_conditions[:, 0]
            model.jacobian_eval = jac_rhs

        # Save CasADi functions for the CasADi solver
        # Note: when we pass to casadi the ode part of the problem must be in explicit
        # form so we pre-multiply by the inverse of the mass matrix
        if model.convert_to_format == "casadi" and isinstance(
            self, pybamm.CasadiSolver
        ):
            mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
            explicit_rhs = mass_matrix_inv @ rhs(
                t_casadi, y_casadi_w_ext, u_casadi_stacked
            )
            model.casadi_rhs = casadi.Function(
                "rhs", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [explicit_rhs]
            )
            model.casadi_algebraic = algebraic

        model.set_up = True
        pybamm.logger.info("Finish solver set-up")

    def set_inputs_and_external(self, model, inputs):
        """
        Set values that are controlled externally, such as external variables and input
        parameters

        Parameters
        ----------
        inputs : dict
            Any input parameters to pass to the model when solving

        """
        model.rhs_eval.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
        model.algebraic_eval.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
        if hasattr(model, "residuals"):
            model.residuals.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
        for evnt in model.events_eval:
            evnt.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)
        if model.jacobian_eval:
            model.jacobian_eval.set_pad_ext_inputs(self.y_pad, self.y_ext, inputs)

    def calculate_consistent_initial_conditions(self, model):
        """
        Calculate consistent initial conditions for the algebraic equations through
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
        y0_guess = model.concatenated_initial_conditions.flatten()
        jac = model.jac_algebraic_eval

        # Split y0_guess into differential and algebraic
        len_rhs = rhs(0, y0_guess).shape[0]
        y0_diff, y0_alg_guess = np.split(y0_guess, [len_rhs])

        def root_fun(y0_alg):
            "Evaluates algebraic using y0_diff (fixed) and y0_alg (changed by algo)"
            y0 = np.concatenate([y0_diff, y0_alg])
            out = algebraic(0, y0)
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

        # Set up
        timer = pybamm.Timer()
        inputs = inputs or {}
        self.y_pad = np.zeros((model.y_length - model.external_start, 1))
        self.set_external_variables(model, external_variables)
        self.set_up(model, inputs)
        set_up_time = timer.time()

        # Solve
        # Set inputs and external
        self.set_inputs_and_external(model, inputs)

        timer.reset()
        pybamm.logger.info("Calling solver")
        solution = self.integrate(model, t_eval, inputs)

        # Assign times
        solution.set_up_time = set_up_time
        solution.solve_time = timer.time()

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

    def step(self, model, dt, npts=2, external_variables=None, inputs=None):
        """
        Step the solution of the model forward by a given time increment. The
        first time this method is called it executes the necessary setup by
        calling `self.set_up(model)`.

        Parameters
        ----------
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
        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            raise pybamm.ModelError("Cannot step empty model")

        # Set timer
        timer = pybamm.Timer()
        inputs = inputs or {}

        if not hasattr(model, "y0"):
            # create a y_pad vector of the correct size:
            self.y_pad = np.zeros((model.y_length - model.external_start, 1))

        self.set_external_variables(model, external_variables)

        # Run set up on first step
        if not hasattr(model, "y0"):
            pybamm.logger.info(
                "Start stepping {} with {}".format(model.name, self.name)
            )
            self.set_up(model, inputs)
            model.t = 0.0
            set_up_time = timer.time()

        else:
            set_up_time = 0

        # Step
        t_eval = np.linspace(model.t, model.t + dt, npts)
        # Set inputs and external
        self.set_inputs_and_external(model, inputs)

        pybamm.logger.info("Calling solver")
        timer.reset()
        solution = self.integrate(model, t_eval, inputs)

        # Assign times
        solution.set_up_time = set_up_time
        solution.solve_time = timer.time()

        # Add model and inputs to solution
        solution.model = model
        solution.inputs = inputs

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, model.events)

        # Set self.t and self.y0 to their values at the final step
        model.t = solution.t[-1]
        model.y0 = solution.y[:, -1]

        # add the external points onto the solution
        full_y = np.zeros((model.y_length, solution.y.shape[1]))
        for i in np.arange(solution.y.shape[1]):
            sol_y = solution.y[:, i]
            sol_y = sol_y[:, np.newaxis]
            full_y[:, i] = add_external(sol_y, self.y_pad, self.y_ext)[:, 0]
        solution.y = full_y

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
        return solution

    def set_external_variables(self, model, external_variables):
        if external_variables is None:
            external_variables = {}

        # load external variables into a state vector
        self.y_ext = np.zeros((model.y_length, 1))
        for var_name, var_vals in external_variables.items():
            var = model.variables[var_name]
            if isinstance(var, pybamm.Concatenation):
                start = var.children[0].y_slices[0].start
                stop = var.children[-1].y_slices[-1].stop
                y_slice = slice(start, stop)

            elif isinstance(var, pybamm.StateVector):
                start = var.y_slices[0].start
                stop = var.y_slices[-1].stop
                y_slice = slice(start, stop)
            else:
                raise pybamm.InputError(
                    """The variable you have inputted is not a StateVector or Concatenation
            of StateVectors. Please check the submodel you have made "external" and
            ensure that the variable you
            are passing in is the variable that is solved for in that submodel"""
                )
            self.y_ext[y_slice] = var_vals

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
            for name, event in events.items():
                y_event = add_external(solution.y_event, self.y_pad, self.y_ext)
                final_event_values[name] = abs(
                    event.evaluate(solution.t_event, y_event)
                )
            termination_event = min(final_event_values, key=final_event_values.get)
            # Add the event to the solution object
            solution.termination = "event: {}".format(termination_event)
            return "the termination event '{}' occurred".format(termination_event)


def add_external(y, y_pad, y_ext):
    """
    Pad the state vector and then add the external variables so that
    it is of the correct shape for evaluate
    """
    if y_pad is not None and y_ext is not None:
        y = np.concatenate([y, y_pad]) + y_ext
    return y


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

        self.y_pad = None
        self.y_ext = None

    def set_pad_ext_inputs(self, y_pad, y_ext, inputs):
        "Set padding, external variables and inputs"
        self.y_pad = y_pad
        self.y_ext = y_ext
        if self.form == "python":
            self.inputs = inputs
        elif self.form == "casadi":
            self.inputs = casadi.vertcat(*[x for x in inputs.values()])

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        if self.name in ["RHS", "algebraic", "residuals"]:
            return self.function(t, y)[:, 0]
        else:
            return self.function(t, y)

    def function(self, t, y):
        if self.form == "casadi":
            return self._function(t, y, self.inputs).full()
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
