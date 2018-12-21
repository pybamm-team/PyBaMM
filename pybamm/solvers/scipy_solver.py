#
# Solver class using Scipy's adaptive time stepper
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import scipy.integrate as it


class ScipySolver(pybamm.BaseSolver):
    def __init__(self, method="BDF", tol=1e-8):
        super().__init__(tol)
        self.method = method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = method

    def integrate(self, derivs, y0, t_eval):
        return it.solve_ivp(
            derivs,
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=self.method,
            rtol=self.tol,
            atol=self.tol,
        )
        # TODO: implement concentration cut-off event
