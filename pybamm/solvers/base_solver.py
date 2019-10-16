#
# Base solver class
#
import pybamm
import numpy as np


class BaseSolver(object):
    """Solve a discretised model.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8).
    """

    def __init__(self, method=None, tol=1e-8):
        self._method = method
        self._tol = tol

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

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
        pybamm.logger.info("Start solving {}".format(model.name))

        # Make sure model isn't empty
        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            raise pybamm.ModelError("Cannot solve empty model")

        # Set up
        timer = pybamm.Timer()
        start_time = timer.time()
        self.set_up(model)
        set_up_time = timer.time() - start_time

        # Solve
        solution, solve_time, termination = self.compute_solution(model, t_eval)

        # Assign times
        solution.solve_time = solve_time
        solution.total_time = timer.time() - start_time
        solution.set_up_time = set_up_time

        pybamm.logger.warning("Finish solving {} ({})".format(model.name, termination))
        pybamm.logger.info(
            "Set-up time: {}, Solve time: {}, Total time: {}".format(
                timer.format(solution.set_up_time),
                timer.format(solution.solve_time),
                timer.format(solution.total_time),
            )
        )
        return solution

    def step(self, model, dt, npts=2):
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

        # Run set up on first step
        if not hasattr(self, "y0"):
            start_time = timer.time()
            self.set_up(model)
            self.t = 0.0
            set_up_time = timer.time() - start_time
        else:
            set_up_time = None

        # Step
        pybamm.logger.info("Start stepping {}".format(model.name))
        t_eval = np.linspace(self.t, self.t + dt, npts)
        solution, solve_time, termination = self.compute_solution(model, t_eval)

        # Assign times
        solution.solve_time = solve_time
        if set_up_time:
            solution.total_time = timer.time() - start_time
            solution.set_up_time = set_up_time

        # Set self.t and self.y0 to their values at the final step
        self.t = solution.t[-1]
        self.y0 = solution.y[:, -1]

        pybamm.logger.info("Finish stepping {} ({})".format(model.name, termination))
        if set_up_time:
            pybamm.logger.info(
                "Set-up time: {}, Step time: {}, Total time: {}".format(
                    timer.format(solution.set_up_time),
                    timer.format(solution.solve_time),
                    timer.format(solution.total_time),
                )
            )
        else:
            pybamm.logger.info(
                "Step time: {}".format(timer.format(solution.solve_time))
            )
        return solution

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

        Raises
        ------
        :class:`pybamm.SolverError`
            If the model contains any algebraic equations (in which case a DAE solver
            should be used instead)

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
                final_event_values[name] = abs(
                    event.evaluate(solution.t_event, solution.y_event)
                )
            termination_event = min(final_event_values, key=final_event_values.get)
            # Add the event to the solution object
            solution.termination = "event: {}".format(termination_event)
            return "the termination event '{}' occurred".format(termination_event)
