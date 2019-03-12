#
# Base solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm
import autograd.numpy as np


class DaeSolver(pybamm.BaseSolver):
    """Solve a discretised model.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance for the solver (default is 1e-8).
    """

    def __init__(self, tol=1e-8):
        super().__init__(tol)

    def solve(self, model, t_eval):
        """Calculate the solution of the model at specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel` (or subclass)
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """

        def residuals(t, y, ydot):
            rhs_eval = model.concatenated_rhs.evaluate(t, y)
            return np.concatenate(
                (
                    rhs_eval - ydot[: rhs_eval.shape[0]],
                    model.concatenated_algebraic.evaluate(t, y),
                )
            )

        y0 = model.concatenated_initial_conditions
        ydot0 = model.concatenated_initial_conditions_ydot

        assert y0.shape == ydot0.shape, pybamm.ModelError(
            "Shape of initial condition y0 {} is different from the shape of initial "
            "condition ydot0 {}".format(y0.shape, ydot0.shape)
        )
        assert y0.shape == residuals(0, y0, ydot0).shape, pybamm.ModelError(
            "Shape of initial condition y0 {} is different from the shape of residual "
            "function {}".format(y0.shape, residuals(0, y0, ydot0).shape)
        )

        self.t, self.y = self.integrate(residuals, y0, ydot0, t_eval)

    def integrate(self, residuals, y0, ydot0, t_eval, jacobian=None, events=None):
        """
        Solve a DAE model defined by residuals with initial conditions y0 and ydot0.

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
            A function that takes in t, y and ydot and returns the Jacobian
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop

        """
        raise NotImplementedError
