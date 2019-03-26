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
        # scikits_odes.sundials.cvode.CV_JacRhsFunction
        # but is OK if you import cvode here?
        from scikits.odes.sundials import cvode

autograd_spec = importlib.util.find_spec("autograd")
if autograd_spec is not None:
    autograd = importlib.util.module_from_spec(autograd_spec)
    autograd_spec.loader.exec_module(autograd)


class ScikitsOdeSolver(pybamm.OdeSolver):
    """Solve a discretised model, using scikits.odes.

    Parameters
    ----------
    method : string, optional
        The method to use in solve_ivp (default is "BDF")
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8). Set as the both reltol and
        abstol in solve_ivp.
    """

    def __init__(self, method="cvode", tol=1e-8):
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

    def integrate(self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None):
        """
        Solve a model defined by dydt with initial conditions y0.

        Parameters
        ----------
        derivs : method
            A function that takes in t and y and returns the time-derivative dydt
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            no Jacobian is provided (default), autograd is used to compute the
            Jacobian. If autograd not installed, the solver will approximate the
            Jacobian. If autograd not installed, the solver will approximate the
            Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).            

        """

        def eqsydot(t, y, return_ydot):
            return_ydot[:] = derivs(t, y)

        def rootfn(t, y, return_root):
            return_root[:] = [event(t, y) for event in events]

        extra_options = {"old_api": False, "rtol": self.tol, "atol": self.tol}

        if jacobian:
            # Put the user-supplied Jacobian into the SUNDIALS Class
            jacfn = JacobianFunctionCV()
            jacfn.set_jacobian(jacobian=jacobian)
            extra_options.update({"jacfn": jacfn})
        elif autograd_spec is None:
            print(
                "autograd is not installed. " "SUNDIALS will approximate the Jacobian."
            )
        else:
            # Calculate the Jacobian using autograd
            jacfn = JacobianFunctionCV()
            jacfn.set_jacobian(derivs=derivs)
            extra_options.update({"jacfn": jacfn})

        if events:
            extra_options.update({"rootfn": rootfn, "nr_rootfns": len(events)})

        ode_solver = scikits_odes.ode(self.method, eqsydot, **extra_options)
        sol = ode_solver.solve(t_eval, y0)

        # return solution, we need to tranpose y to match scipy's ivp interface
        return sol.values.t, np.transpose(sol.values.y)


class JacobianFunctionCV(cvode.CV_JacRhsFunction):
    def set_jacobian(self, jacobian=None, derivs=None):
        """
        Sets the user supplied Jacobian function for the ODE model. If no
        Jacobian is supplied, the user must supply the function derivs so
        that the Jacobian may be calculated using autograd.

        Parameters
        ----------
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian
        derivs : method, optional
            A function that takes in t and y and returns the time-derivative dydt

        """
        if jacobian:
            self.jacobian = jacobian
        else:
            self.jacobian = autograd.jacobian(derivs, 1)

    def evaluate(self, t, y, fy, return_jacobian):
        return_jacobian[:][:] = self.jacobian(t, y)
        return 0
