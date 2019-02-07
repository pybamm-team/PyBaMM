#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import scipy.integrate as it


class ScipySolver(pybamm.OdeSolver):
    """Solve a discretised model, using scipy.integrate.solve_ivp.

    Parameters
    ----------
    method : string, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, method="BDF", tol=1e-8):
        super().__init__(tol)
        self.method = method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    def integrate(self, derivs, y0, t_eval):
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

        Returns
        -------
        object
            An object containing the times and values of the solution, as well as
            various diagnostic messages.
        """
        sol = it.solve_ivp(
            derivs,
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=self.method,
            rtol=self.tol,
            atol=self.tol,
        )
        # TODO: implement concentration cut-off event
        return sol.t, sol.y
