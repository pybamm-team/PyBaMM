#
# Base solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm


class OdeSolver(pybamm.BaseSolver):
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

        def dydt(t, y):
            return model.concatenated_rhs.evaluate(t, y)

        events = [lambda t, y: event.evaluate(t, y) for event in model.events]

        y0 = model.concatenated_initial_conditions
        self.t, self.y = self.integrate(dydt, y0, t_eval, events=events)

    def integrate(self, derivs, y0, t_eval, events=None, mass_matrix=None):
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
        mass_matrix : :class:`pybamm.Matrix`
            The (sparse) mass matrix for the chosen spatial method.
        """
        raise NotImplementedError
