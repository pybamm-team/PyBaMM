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
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, method="BDF", tol=1e-8):
        super().__init__(method, tol)

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
        extra_options = {"rtol": self.tol, "atol": self.tol}

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
            **extra_options
        )

        if sol.success:
            return pybamm.Solution(sol.t, sol.y)
        else:
            raise pybamm.SolverError(sol.message)
