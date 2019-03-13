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
            A function that takes in t, y and ydot and returns the Jacobian. If
            no Jacobian provided (default), autograd is used to compute the
            Jacobian. If autograd not installed, the solver will approximate the
            Jacobian (see SUNDIALS_ documentation).
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop

        .. _SUNDIALS: https://computation.llnl.gov/sites/default/files/public/ida_guide.pdf

        """

        def eqsres(t, y, ydot, return_residuals):
            return_residuals[:] = residuals(t, y, ydot)

        def rootfn(t, y, ydot, return_root):
            return_root[:] = [event(t, y) for event in events]

        extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol}

        if jacobian is None:
            if autograd_spec is None:
                print(
                    "autograd is not installed. "
                    "SUNDIALS will approximate the Jacobian."
                )
            else:
                # Set automatic jacobian function
                self.set_auto_jac(residuals)

                # Constant mass matrix
                mass_matrix = -self.jacobian_ydot(0.0, y0, ydot0)
                algebraic_vars_idx = np.where(~mass_matrix.any(axis=1))[0]

                # Can't pass self.jacobian_y into jacfn
                # - is there a better way of doing this?
                jac_y = self.jacobian_y

                def jacfn(self, t, y, ydot, cj, return_jacobian):
                    return_jacobian[:][:] = jac_y(t, y, ydot) - cj * mass_matrix

                extra_options.update(
                    {"jacfn": jacfn, "algebraic_vars_idx": algebraic_vars_idx}
                )

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        dae_solver = scikits_odes.dae(self.method, eqsres, **extra_options)
        sol = dae_solver.solve(t_eval, y0, ydot0)

        # return solution, we need to tranpose y to match scipy's interface
        return sol.values.t, np.transpose(sol.values.y)

    def set_auto_jac(self, residuals):
        """
        Sets the Jacobian function for the DAE model using autograd

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations

        """
        self.jacobian_ydot = autograd.jacobian(residuals, 2)
        self.jacobian_y = autograd.jacobian(residuals, 1)
