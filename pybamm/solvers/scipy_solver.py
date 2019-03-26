#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import scipy.integrate as it
import numpy as np
import importlib

autograd_spec = importlib.util.find_spec("autograd")
if autograd_spec is not None:
    autograd = importlib.util.module_from_spec(autograd_spec)
    autograd_spec.loader.exec_module(autograd)


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

    def integrate(self, derivs, y0, t_eval, events=None, mass_matrix=None, jacobian=None):
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
        mass_matrix : array_like
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t and y and returns the Jacobian. If
            no Jacobian is provided (default), autograd is used to compute the
            Jacobian. If autograd not installed, the solver will approximate the
            Jacobian.
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
            if jacobian is None:
                if autograd_spec is None:
                    print(
                        "autograd is not installed. "
                        "scipy will approximate the Jacobian."
                    )
                else:
                    # Set automatic jacobian function
                    self.set_auto_jac(derivs)

                    def jacobian(t, y):
                        return self.jacobian(t, y)

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

        return sol.t, sol.y

    def set_auto_jac(self, derivs):
        """
        Sets the Jacobian function for the ODE model using autograd

        Parameters
        ----------
        derivs : method
            A function that takes in t and y and returns the time-derivative dydt

        """
        self.jacobian = autograd.jacobian(derivs, 1)
