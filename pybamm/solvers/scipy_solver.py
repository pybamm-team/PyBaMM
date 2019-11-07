#
# Solver class using Scipy's adaptive time stepper
#
import pybamm

import scipy.integrate as it
import numpy as np


class ScipySolver(pybamm.OdeSolver):
    """Solve a discretised model, using scipy.integrate.solve_ivp.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    """

    def __init__(self, method="BDF", rtol=1e-6, atol=1e-6):
        super().__init__(method, rtol, atol)
        self.name = "Scipy solver ({})".format(method)

    def integrate(
        self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        derivs : method
            A function that takes in t (size (1,)), y (size (n,))
            and returns the time-derivative dydt (size (n,))
        y0 : :class:`numpy.array`, size (n,)
            The initial conditions
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
        Returns
        -------
        object
            An object containing the times and values of the solution, as well as
            various diagnostic messages.

        """
        extra_options = {"rtol": self.rtol, "atol": self.atol}

        # check for user-supplied Jacobian
        implicit_methods = ["Radau", "BDF", "LSODA"]
        if np.any([self.method in implicit_methods]):
            if jacobian:
                extra_options.update({"jac": jacobian})

        # make events terminal so that the solver stops when they are reached
        if events:
            for event in events:
                event.terminal = True
            extra_options.update({"events": events})

        sol = it.solve_ivp(
            derivs,
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
