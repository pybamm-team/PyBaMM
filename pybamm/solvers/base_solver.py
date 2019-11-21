#
# Base solver class
#
import pybamm
import numpy as np


class BaseSolver(object):
    """Solve a discretised model.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    """

    def __init__(self, method=None, rtol=1e-6, atol=1e-6):
        self._method = method
        self._rtol = rtol
        self._atol = atol
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

    def solve(self, model, t_eval):
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
        if model.convert_to_format == "casadi" or isinstance(self, pybamm.CasadiSolver):
            self.set_up_casadi(model)
        else:
            self.set_up(model)
        set_up_time = timer.time() - start_time

        # Solve
        solution, solve_time, termination = self.compute_solution(model, t_eval)

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

    def step(self, model, dt, npts=2, log=True, external_variables=None):
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
                self.set_up_casadi(model)
            else:
                pybamm.logger.debug(
                    "Start stepping {} with {}".format(model.name, self.name)
                )
                self.set_up(model)
            self.t = 0.0
            set_up_time = timer.time()

        else:
            set_up_time = 0

        # Step
        t_eval = np.linspace(self.t, self.t + dt, npts)
        solution, solve_time, termination = self.compute_solution(model, t_eval)

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

    def compute_solution(self, model, t_eval):
        """Calculate the solution of the model at specified times. Note: this
        does *not* execute the solver setup.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        raise NotImplementedError

    def set_up(self, model):
        """Unpack model, perform checks, simplify and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions

        """
        raise NotImplementedError

    def set_up_casadi(self, model):
        """Convert model to casadi format and use their inbuilt functionalities.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions

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
