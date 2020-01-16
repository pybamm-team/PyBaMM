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

        # create simplified rhs, algebraic and event expressions
        concatenated_rhs = model.concatenated_rhs
        concatenated_algebraic = model.concatenated_algebraic
        events = model.events
        y0 = model.concatenated_initial_conditions
        y0 = add_external(y0, self.y_pad, self.y_ext)

        if model.convert_to_format != "casadi":
            if model.use_simplify:
                # set up simplification object, for re-use of dict
                simp = pybamm.Simplification()
                pybamm.logger.info("Simplifying RHS")
                concatenated_rhs = simp.simplify(concatenated_rhs)
                pybamm.logger.info("Simplifying algebraic")
                concatenated_algebraic = simp.simplify(concatenated_algebraic)
                pybamm.logger.info("Simplifying events")
                events = {name: simp.simplify(event) for name, event in events.items()}

            if model.use_jacobian:
                # Create Jacobian from concatenated rhs and algebraic
                y = pybamm.StateVector(
                    slice(0, np.size(model.concatenated_initial_conditions))
                )
                # set up Jacobian object, for re-use of dict
                jacobian = pybamm.Jacobian()
                pybamm.logger.info("Calculating jacobian")
                jac_rhs = jacobian.jac(concatenated_rhs, y)
                jac_algebraic = jacobian.jac(concatenated_algebraic, y)
                jac = pybamm.SparseStack(jac_rhs, jac_algebraic)
                model.jacobian = jac
                model.jacobian_rhs = jac_rhs
                model.jacobian_algebraic = jac_algebraic

                if model.use_simplify:
                    pybamm.logger.info("Simplifying jacobian")
                    jac_algebraic = simp.simplify(jac_algebraic)
                    jac = simp.simplify(jac)

                if model.convert_to_format == "python":
                    pybamm.logger.info("Converting jacobian to python")
                    jac_algebraic = pybamm.EvaluatorPython(jac_algebraic)
                    jac = pybamm.EvaluatorPython(jac)

            if model.convert_to_format == "python":
                pybamm.logger.info("Converting RHS to python")
                concatenated_rhs = pybamm.EvaluatorPython(concatenated_rhs)
                pybamm.logger.info("Converting algebraic to python")
                concatenated_algebraic = pybamm.EvaluatorPython(concatenated_algebraic)
                pybamm.logger.info("Converting events to python")
                events = {
                    name: pybamm.EvaluatorPython(event) for name, event in events.items()
                }
            
            concatenated_rhs_fn = concatenated_rhs.evaluate
            concatenated_algebraic_fn = concatenated_algebraic.evaluate

        elif model.convert_to_format == "casadi":
            # Convert model attributes to casadi
            t_casadi = casadi.MX.sym("t")

            y_diff = casadi.MX.sym(
                "y_diff", len(concatenated_rhs.evaluate(0, y0, inputs))
            )
            y_alg = casadi.MX.sym(
                "y_alg", len(concatenated_algebraic.evaluate(0, y0, inputs))
            )
            y_casadi = casadi.vertcat(y_diff, y_alg)
            if self.y_pad is not None:
                y_ext = casadi.MX.sym("y_ext", len(self.y_pad))
                y_casadi_w_ext = casadi.vertcat(y_casadi, y_ext)
            else:
                y_casadi_w_ext = y_casadi
            u_casadi = {name: casadi.MX.sym(name) for name in inputs.keys()}

            pybamm.logger.info("Converting RHS to CasADi")
            concatenated_rhs = model.concatenated_rhs.to_casadi(
                t_casadi, y_casadi_w_ext, u_casadi
            )
            pybamm.logger.info("Converting algebraic to CasADi")
            concatenated_algebraic = model.concatenated_algebraic.to_casadi(
                t_casadi, y_casadi_w_ext, u_casadi
            )
            all_states = casadi.vertcat(concatenated_rhs, concatenated_algebraic)
            pybamm.logger.info("Converting events to CasADi")
            events_fn = {
                name: event.to_casadi(t_casadi, y_casadi_w_ext, u_casadi)
                for name, event in model.events.items()
            }

            # Create functions to evaluate rhs and algebraic
            u_casadi_stacked = casadi.vertcat(*[u for u in u_casadi.values()])
            concatenated_rhs_fn = casadi.Function(
                "rhs", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [concatenated_rhs]
            )
            concatenated_algebraic_fn = casadi.Function(
                "algebraic",
                [t_casadi, y_casadi_w_ext, u_casadi_stacked],
                [concatenated_algebraic],
            )
            all_states_fn = casadi.Function(
                "all", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [all_states]
            )

            if model.use_jacobian:

                pybamm.logger.info("Calculating jacobian")
                casadi_jac = casadi.jacobian(all_states, y_casadi)
                casadi_jac_fn = casadi.Function(
                    "jacobian", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [casadi_jac]
                )
                casadi_jac_alg = casadi.jacobian(concatenated_algebraic, y_casadi)
                casadi_jac_alg_fn = casadi.Function(
                    "jacobian",
                    [t_casadi, y_casadi_w_ext, u_casadi_stacked],
                    [casadi_jac_alg],
                )

        if model.use_jacobian:
            jacobian = Jacobian(casadi_jac_fn, form="casadi")
            jacobian_alg = Jacobian(casadi_jac_alg_fn, form="casadi")
            jacobian_alg.set_pad_ext(self.y_pad, self.y_ext)
            jacobian_alg.set_inputs(inputs)

        else:
            jacobian = None
            jacobian_alg = None

        # Calculate consistent initial conditions for the algebraic equations
        rhs = Rhs(concatenated_rhs.evaluate)
        algebraic = Algebraic(concatenated_algebraic.evaluate)

        rhs.set_pad_ext(self.y_pad, self.y_ext)
        rhs.set_inputs(inputs)
        algebraic.set_pad_ext(self.y_pad, self.y_ext)
        algebraic.set_inputs(inputs)

        if len(model.algebraic) > 0:
            y0 = self.calculate_consistent_initial_conditions(
                rhs,
                algebraic,
                model.concatenated_initial_conditions[:, 0],
                jacobian_alg,
            )
        else:
            # can use DAE solver to solve ODE model
            y0 = model.concatenated_initial_conditions[:, 0]

        # Create event-dependent function to evaluate events
        def get_event_class(event):
            return EvalEvent(event.evaluate)

        # Add the solver attributes
        self.y0 = y0
        self.rhs = rhs
        self.algebraic = algebraic
        self.residuals = Residuals(
            model, concatenated_rhs.evaluate, concatenated_algebraic.evaluate
        )
        self.events = events
        self.event_funs = [get_event_class(event) for event in events.values()]
        self.jacobian = jacobian

        # Save CasADi functions for the CasADi solver
        # Note: when we pass to casadi the ode part of the problem must be in explicit
        # form so we pre-multiply by the inverse of the mass matrix
        if model.convert_to_format == "casadi" and isinstance(self, pybamm.CasadiSolver):
            mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
            explicit_rhs = mass_matrix_inv @ concatenated_rhs
            self.casadi_rhs = casadi.Function(
                "rhs", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [explicit_rhs]
            )
            self.casadi_algebraic = concatenated_algebraic_fn

        pybamm.logger.info("Finish solver set-up")

    def set_up_casadi(self, model, inputs=None):
        """Convert model to casadi format and use their inbuilt functionalities.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        """
        inputs = inputs or {}

        # Convert model attributes to casadi
        t_casadi = casadi.MX.sym("t")
        y0 = model.concatenated_initial_conditions
        y0 = add_external(y0, self.y_pad, self.y_ext)

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

        pybamm.logger.info("Converting RHS to CasADi")
        concatenated_rhs = model.concatenated_rhs.to_casadi(
            t_casadi, y_casadi_w_ext, u_casadi
        )
        pybamm.logger.info("Converting algebraic to CasADi")
        concatenated_algebraic = model.concatenated_algebraic.to_casadi(
            t_casadi, y_casadi_w_ext, u_casadi
        )
        all_states = casadi.vertcat(concatenated_rhs, concatenated_algebraic)
        pybamm.logger.info("Converting events to CasADi")
        casadi_events = {
            name: event.to_casadi(t_casadi, y_casadi_w_ext, u_casadi)
            for name, event in model.events.items()
        }

        # Create functions to evaluate rhs and algebraic
        u_casadi_stacked = casadi.vertcat(*[u for u in u_casadi.values()])
        concatenated_rhs_fn = casadi.Function(
            "rhs", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [concatenated_rhs]
        )
        concatenated_algebraic_fn = casadi.Function(
            "algebraic",
            [t_casadi, y_casadi_w_ext, u_casadi_stacked],
            [concatenated_algebraic],
        )
        all_states_fn = casadi.Function(
            "all", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [all_states]
        )

        if model.use_jacobian:

            pybamm.logger.info("Calculating jacobian")
            casadi_jac = casadi.jacobian(all_states, y_casadi)
            casadi_jac_fn = casadi.Function(
                "jacobian", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [casadi_jac]
            )
            casadi_jac_alg = casadi.jacobian(concatenated_algebraic, y_casadi)
            casadi_jac_alg_fn = casadi.Function(
                "jacobian",
                [t_casadi, y_casadi_w_ext, u_casadi_stacked],
                [casadi_jac_alg],
            )

            jacobian = Jacobian(casadi_jac_fn, form="casadi")
            jacobian_alg = Jacobian(casadi_jac_alg_fn, form="casadi")
            jacobian_alg.set_pad_ext(self.y_pad, self.y_ext)
            jacobian_alg.set_inputs(inputs)

        else:
            jacobian = None
            jacobian_alg = None

        rhs = Rhs(concatenated_rhs_fn, form="casadi")
        algebraic = Algebraic(concatenated_algebraic_fn, form="casadi")

        rhs.set_pad_ext(self.y_pad, self.y_ext)
        rhs.set_inputs(inputs)
        algebraic.set_pad_ext(self.y_pad, self.y_ext)
        algebraic.set_inputs(inputs)

        if len(model.algebraic) > 0:

            y0 = self.calculate_consistent_initial_conditions(
                rhs,
                algebraic,
                model.concatenated_initial_conditions[:, 0],
                jacobian_alg,
            )
        else:
            # can use DAE solver to solve ODE model
            y0 = model.concatenated_initial_conditions[:, 0]

        # Create event-dependent function to evaluate events
        def get_event_class(event):
            casadi_event_fn = casadi.Function(
                "event", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [event]
            )
            return EvalEvent(casadi_event_fn, form="casadi")

        # Add the solver attributes
        # Note: these are the (possibly) converted to python version rhs, algebraic
        # etc. The expression tree versions of these are attributes of the model
        self.y0 = y0
        self.rhs = rhs
        self.algebraic = algebraic
        self.residuals = Residuals(model, all_states_fn, form="casadi")
        self.events = model.events
        self.event_funs = [get_event_class(event) for event in casadi_events.values()]
        self.jacobian = jacobian

        # Save CasADi functions for the CasADi solver
        # Note: when we pass to casadi the ode part of the problem must be in explicit
        # form so we pre-multiply by the inverse of the mass matrix
        if isinstance(self, pybamm.CasadiSolver):
            mass_matrix_inv = casadi.MX(model.mass_matrix_inv.entries)
            explicit_rhs = mass_matrix_inv @ concatenated_rhs
            self.casadi_rhs = casadi.Function(
                "rhs", [t_casadi, y_casadi_w_ext, u_casadi_stacked], [explicit_rhs]
            )
            self.casadi_algebraic = concatenated_algebraic_fn

        pybamm.logger.info("Finish solver set-up")

    def set_inputs_and_external(self, inputs):
        """
        Set values that are controlled externally, such as external variables and input
        parameters

        Parameters
        ----------
        inputs : dict
            Any input parameters to pass to the model when solving

        """
        self.rhs.set_pad_ext(self.y_pad, self.y_ext)
        self.rhs.set_inputs(inputs)
        self.algebraic.set_pad_ext(self.y_pad, self.y_ext)
        self.algebraic.set_inputs(inputs)
        self.residuals.set_pad_ext(self.y_pad, self.y_ext)
        self.residuals.set_inputs(inputs)
        for evnt in self.event_funs:
            evnt.set_pad_ext(self.y_pad, self.y_ext)
            evnt.set_inputs(inputs)
        if self.jacobian:
            self.jacobian.set_pad_ext(self.y_pad, self.y_ext)
            self.jacobian.set_inputs(inputs)

    def calculate_consistent_initial_conditions(
        self, rhs, algebraic, y0_guess, jac=None
    ):
        """
        Calculate consistent initial conditions for the algebraic equations through
        root-finding

        Parameters
        ----------
        rhs : method
            Function that takes in t and y and returns the value of the differential
            equations
        algebraic : method
            Function that takes in t and y and returns the value of the algebraic
            equations
        y0_guess : array-like
            Array of the user's guess for the initial conditions, used to initialise
            the root finding algorithm
        jac : method
            Function that takes in t and y and returns the value of the jacobian for the
            algebraic equations

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations)
        """
        pybamm.logger.info("Start calculating consistent initial conditions")

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

    def solve(self, model, t_eval, inputs=None):
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
        start_time = timer.time()
        inputs = inputs or {}
        if model.convert_to_format == "casadi" or isinstance(self, pybamm.CasadiSolver):
            self.set_up_casadi(model, inputs)
        else:
            self.set_up(model, inputs)
        set_up_time = timer.time() - start_time

        # Solve
        solution, solve_time, termination = self.compute_solution(
            model, t_eval, inputs=inputs
        )

        # Assign times
        solution.solve_time = solve_time
        solution.set_up_time = set_up_time

        pybamm.logger.info("Finish solving {} ({})".format(model.name, termination))
        pybamm.logger.info(
            "Set-up time: {}, Solve time: {}, Total time: {}".format(
                timer.format(solution.set_up_time),
                timer.format(solution.solve_time),
                timer.format(solution.total_time),
            )
        )
        return solution

    def step(self, model, dt, npts=2, log=True, external_variables=None, inputs=None):
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

        if not hasattr(self, "y0"):
            # create a y_pad vector of the correct size:
            self.y_pad = np.zeros((model.y_length - model.external_start, 1))

        self.set_external_variables(model, external_variables)

        # Run set up on first step
        if not hasattr(self, "y0"):
            pybamm.logger.info(
                "Start stepping {} with {}".format(model.name, self.name)
            )

            if model.convert_to_format == "casadi" or isinstance(
                self, pybamm.CasadiSolver
            ):
                self.set_up_casadi(model, inputs)
            else:
                pybamm.logger.debug(
                    "Start stepping {} with {}".format(model.name, self.name)
                )
                self.set_up(model, inputs)
            self.t = 0.0
            set_up_time = timer.time()

        else:
            set_up_time = 0

        # Step
        t_eval = np.linspace(self.t, self.t + dt, npts)
        solution, solve_time, termination = self.compute_solution(model, t_eval, inputs)

        # Set self.t and self.y0 to their values at the final step
        self.t = solution.t[-1]
        self.y0 = solution.y[:, -1]

        # add the external points onto the solution
        full_y = np.zeros((model.y_length, solution.y.shape[1]))
        for i in np.arange(solution.y.shape[1]):
            sol_y = solution.y[:, i]
            sol_y = sol_y[:, np.newaxis]
            full_y[:, i] = add_external(sol_y, self.y_pad, self.y_ext)[:, 0]
        solution.y = full_y

        # Assign times
        solution.solve_time = solve_time
        solution.set_up_time = set_up_time

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

    def compute_solution(self, model, t_eval, inputs=None):
        """Calculate the solution of the model at specified times. Note: this
        does *not* execute the solver setup.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        """
        raise NotImplementedError

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

    def __init__(self, function, form="python"):
        self.function = function
        self.form = form

        self.y_pad = None
        self.y_ext = None
        self.inputs = {}
        self.inputs_casadi = casadi.DM()

    def set_pad_ext(self, y_pad, y_ext):
        self.y_pad = y_pad
        self.y_ext = y_ext

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.inputs_casadi = casadi.vertcat(*[x for x in inputs.values()])

    def __call__(self, t, y):
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        return self.function(t, y)

    def function(self, t, y):
        if self.form == "python":
            return self.function(t, y, self.inputs, known_evals={})[0][:, 0]
        elif self.form == "casadi":
            return self.function(t, y, self.inputs_casadi).full()[:, 0]


class Residuals(SolverCallable):
    "Returns information about residuals at time t and state y"

    def __init__(self, model, concatenated_rhs_fn, concatenated_algebraic_fn):
        self.model = model
        self.concatenated_rhs_fn = concatenated_rhs_fn
        self.concatenated_algebraic_fn = concatenated_algebraic_fn
        self.mass_matrix = model.mass_matrix.entries

    def __call__(self, t, y, ydot):
        pybamm.logger.debug(
            "Evaluating residuals for {} at t={}".format(self.model.name, t)
        )
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        rhs_eval, known_evals = self.concatenated_rhs_fn(
            t, y, self.inputs, known_evals={}
        )
        # reuse known_evals
        alg_eval = self.concatenated_algebraic_fn(
            t, y, self.inputs, known_evals=known_evals
        )[0]
        # turn into 1D arrays
        rhs_eval = rhs_eval[:, 0]
        alg_eval = alg_eval[:, 0]
        return np.concatenate([rhs_eval, alg_eval]) - self.mass_matrix @ ydot


class ResidualsCasadi(Residuals):
    "Returns information about residuals at time t and state y, with CasADi"

    def __init__(self, model, all_states_fn):
        self.model = model
        self.all_states_fn = all_states_fn
        self.mass_matrix = model.mass_matrix.entries

    def __call__(self, t, y, ydot):
        pybamm.logger.debug(
            "Evaluating residuals for {} at t={}".format(self.model.name, t)
        )
        y = y[:, np.newaxis]
        y = add_external(y, self.y_pad, self.y_ext)
        states_eval = self.all_states_fn(t, y, self.inputs_casadi).full()[:, 0]
        return states_eval - self.mass_matrix @ ydot


class EvalEvent(SolverCallable):
    "Returns information about events at time t and state y"

    def function(self, t, y):
        if self.form == "python":
            return self.event_fn(t, y, self.inputs)
        elif self.form == "casadi":
            return self.event_fn(t, y, self.inputs_casadi)


class Jacobian(SolverCallable):
    "Returns information about the jacobian at time t and state y"

    def function(self, t, y):
        if self.form == "python":
            return self.function(t, y, self.inputs, known_evals={})[0]
        elif self.form == "casadi":
            return self.function(t, y, self.inputs_casadi)

