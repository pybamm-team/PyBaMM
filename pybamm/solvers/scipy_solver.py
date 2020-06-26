#
# Solver class using Scipy's adaptive time stepper
#
import casadi
import pybamm

import scipy.integrate as it
import numpy as np


class ScipySolver(pybamm.BaseSolver):
    """Solve a discretised model, using scipy.integrate.solve_ivp.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    extra_options : dict, optional
        Any options to pass to the solver.
        Please consult `SciPy documentation <https://tinyurl.com/yafgqg9y>`_ for
        details.
    """

    def __init__(self, method="BDF", rtol=1e-6, atol=1e-6, extra_options=None):
        super().__init__(method, rtol, atol)
        self.ode_solver = True
        self.extra_options = extra_options or {}
        self.name = "Scipy solver ({})".format(method)
        pybamm.citations.register("virtanen2020scipy")

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        object
            An object containing the times and values of the solution, as well as
            various diagnostic messages.

        """
        if model.convert_to_format == "casadi":
            inputs = casadi.vertcat(*[x for x in inputs.values()])

        extra_options = {**self.extra_options, "rtol": self.rtol, "atol": self.atol}

        # Initial conditions
        y0 = model.y0
        if isinstance(y0, casadi.DM):
            y0 = y0.full().flatten()

        # check for user-supplied Jacobian
        implicit_methods = ["Radau", "BDF", "LSODA"]
        if np.any([self.method in implicit_methods]):
            if model.jacobian_eval:
                extra_options.update(
                    {"jac": lambda t, y: model.jacobian_eval(t, y, inputs)}
                )

        # make events terminal so that the solver stops when they are reached
        if model.terminate_events_eval:

            def event_wrapper(event):
                def event_fn(t, y):
                    return event(t, y, inputs)

                event_fn.terminal = True
                return event_fn

            events = [event_wrapper(event) for event in model.terminate_events_eval]
            extra_options.update({"events": events})

        sol = it.solve_ivp(
            lambda t, y: model.rhs_eval(t, y, inputs),
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=self.method,
            dense_output=True,
            **extra_options
        )

        if sol.success:
            # Set the reason for termination
            if sol.message == "A termination event occurred.":
                termination = "event"
                t_event = []
                for time in sol.t_events:
                    if time.size > 0:
                        t_event = np.append(t_event, np.max(time))
                t_event = np.array([np.max(t_event)])
                y_event = sol.sol(t_event)
            elif sol.message.startswith("The solver successfully reached the end"):
                termination = "final time"
                t_event = None
                y_event = np.array(None)
            return pybamm.Solution(sol.t, sol.y, t_event, y_event, termination)
        else:
            raise pybamm.SolverError(sol.message)
