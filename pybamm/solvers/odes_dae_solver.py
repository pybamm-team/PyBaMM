#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import importlib
scikits_odes = importlib.util.find_spec("scikits")
if scikits_odes is not None:
    scikits_odes = importlib.util.find_spec("scikits.odes")


class OdesDaeSolver(pybamm.DaeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : string, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, tol=1e-8):
        if scikits_odes is None:
            raise ImportError("Error: scikits.odes is not installed, "
                              "please install via \"pip install scikits.odes\"")
        super().__init__(tol)
        self.method = method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    def integrate(self, residuals, y0, ydot0, t_eval):
        """
        Solve a DAE model defined by residuals with initial conditions y0 and ydot_0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        def eqsres(t, y, ydot, return_residuals):
            return_residuals = residuals(t, y, ydot)

        extra_options = {
            'old_api': False,
            'rtol': self.tol,
            'atol': self.tol,
        }

        dae_solver = scikits_odes.dae(self.method, y0, **extra_options)
        sol = dae_solver.solve(t_eval, y0, ydot0)
        return sol.t, sol.y
