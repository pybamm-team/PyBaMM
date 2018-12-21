#
# Base solver class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class BaseSolver(object):
    """Solve a discretised model.

    Parameters
    ----------
    tolerance : float, optional
        The tolerance for the solver.
    """

    def __init__(self, tol=1e-8):
        self._tol = tol

        # Solutions
        self._t = None
        self._y = None

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def solve(self, model, t_eval):
        def dydt(t, y):
            return model.rhs.evaluate(t, y)

        y0 = model.initial_conditions
        sol = self.integrate(dydt, y0, t_eval)

        self.t = sol.t
        self.y = sol.y

    def integrate(self, derivs, y0, t_eval, options={}):
        raise NotImplementedError
