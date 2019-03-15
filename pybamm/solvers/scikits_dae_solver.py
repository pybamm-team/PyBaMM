#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import autograd.numpy as np
import importlib

scikits_odes_spec = importlib.util.find_spec("scikits")
if scikits_odes_spec is not None:
    scikits_odes_spec = importlib.util.find_spec("scikits.odes")
    if scikits_odes_spec is not None:
        scikits_odes = importlib.util.module_from_spec(scikits_odes_spec)
        scikits_odes_spec.loader.exec_module(scikits_odes)
        # NOTE: Gives error module 'scikits.odes' has no attribute 'sundials'
        # if you try to make the Jacobian class using
        # scikits_odes.sundials.ida.IDA_JacRhsFunction
        # but is OK if you import ida here?
        from scikits.odes.sundials import ida

autograd_spec = importlib.util.find_spec("autograd")
if autograd_spec is not None:
    autograd = importlib.util.module_from_spec(autograd_spec)
    autograd_spec.loader.exec_module(autograd)


class ScikitsDaeSolver(pybamm.DaeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : string, optional
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

    def integrate(self, residuals, y0, ydot0, t_eval, jacobian=None, events=None):
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
        jacobian : method, optional
            A function that takes in t, y, ydot and cj and returns the Jacobian. If
            no Jacobian is provided (default), autograd is used to compute the
            Jacobian. If autograd not installed, the solver will approximate the
            Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop

        """

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot)

        def rootfn(t, y, ydot, return_root):
            return_root[:] = [event(t, y) for event in events]

        extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol}

        if jacobian:
            # Put the user-supplied Jacobian into the SUNDIALS Class
            jacfn = JacobianFunctionIDA()
            jacfn.set_jacobian(jacobian)
            extra_options.update({"jacfn": jacfn})
        elif autograd_spec is None:
            print(
                "autograd is not installed. " "SUNDIALS will approximate the Jacobian."
            )
        else:
            # Calculate the Jacobian using autograd
            jacfn = AutoJacobianFunctionIDA()
            jacfn.set_jacobian(residuals, y0)
            algebraic_vars_idx = np.where(~jacfn.mass_matrix.any(axis=1))[0]
            extra_options.update(
                {"jacfn": jacfn, "algebraic_vars_idx": algebraic_vars_idx}
            )

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        sol = dae_solver.solve(t_eval, y0, ydot0)

        # return solution, we need to tranpose y to match scipy's interface
        return sol.values.t, np.transpose(sol.values.y)


class JacobianFunctionIDA(ida.IDA_JacRhsFunction):
    def set_jacobian(self, jacobian):
        """
        Sets the user supplied Jacobian function for the DAE model.

        Parameters
        ----------
        jacobian : method
            A function that takes in t, y, ydot and cj and returns the Jacobian.

        """
        self.jacobian = jacobian

    def evaluate(self, t, y, ydot, residuals, cj, return_jacobian):
        return_jacobian[:][:] = self.jacobian(t, y, ydot, cj)
        return 0


class AutoJacobianFunctionIDA(ida.IDA_JacRhsFunction):
    def set_jacobian(self, residuals, y0):
        """
        Calculates the Jacobian function for the DAE model using autograd.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions

        """
        self.jacobian_y = autograd.jacobian(residuals, 1)
        self.jacobian_ydot = autograd.jacobian(residuals, 2)
        # Note: PyBaMM models written so that mass matrix is constant
        self.mass_matrix = -self.jacobian_ydot(0.0, y0, y0)

    def evaluate(self, t, y, ydot, residuals, cj, return_jacobian):
        return_jacobian[:][:] = self.jacobian_y(t, y, ydot) - cj * self.mass_matrix
        return 0
