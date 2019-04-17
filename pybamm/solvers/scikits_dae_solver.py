#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import importlib
from scipy.sparse import issparse

scikits_odes_spec = importlib.util.find_spec("scikits")
if scikits_odes_spec is not None:
    scikits_odes_spec = importlib.util.find_spec("scikits.odes")
    if scikits_odes_spec is not None:
        scikits_odes = importlib.util.module_from_spec(scikits_odes_spec)
        scikits_odes_spec.loader.exec_module(scikits_odes)
        from scikits.odes.sundials import ida


class ScikitsDaeSolver(pybamm.DaeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : str, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, method="ida", tol=1e-8):
        if scikits_odes_spec is None:
            raise ImportError("scikits.odes is not installed")

        super().__init__(tol)
        self._method = method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    def integrate(
        self, residuals, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        """

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot)

        def rootfn(t, y, ydot, return_root):
            return_root[:] = [event(t, y) for event in events]

        extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol}

        if jacobian:
            # Put the user-supplied Jacobian into the SUNDIALS Class
            jacfn = JacobianFunctionIDA()
            jacfn.set_jacobian(mass_matrix, jacobian)
            extra_options.update({"jacfn": jacfn})

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        # set up and solve
        dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        sol = dae_solver.solve(t_eval, y0, ydot0)

        # return solution, we need to tranpose y to match scipy's interface
        return sol.values.t, np.transpose(sol.values.y)


class JacobianFunctionIDA(ida.IDA_JacRhsFunction):
    def set_jacobian(self, mass_matrix, jacobian):
        """
        Sets the user supplied mass matrix and Jacobian function for the DAE model.

        Parameters
        ----------
        mass_matrix : array_like
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method
            A function that takes in t and y and returns the Jacobian.

        """
        self.mass_matrix = mass_matrix
        self.jacobian = jacobian

    def evaluate(self, t, y, ydot, residuals, cj, return_jacobian):
        # scikits_odes requires the full (dense) jacobian
        jac_eval = self.jacobian(t, y) - cj * self.mass_matrix
        if issparse(jac_eval):
            return_jacobian[:][:] = jac_eval.toarray()
        else:
            return_jacobian[:][:] = jac_eval

        return 0
