#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class BaseModel(object):
    """Base model class for other models to extend.

    Parameters
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to
        A dictionary for testing the convergence of the numerical solution:
            * {} (default): We are not running in test mode, use built-ins.
            * {'inits': dict of initial conditions,
               'bcs': dict of boundary conditions,
               'sources': dict of source terms
               }: To be used for testing convergence to an exact solution.
    """

    def __init__(self):
        self._rhs = {}
        self._initial_conditions = {}
        self._boundary_conditions = {}

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, rhs):
        self._rhs = rhs

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        self._initial_conditions = initial_conditions

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def initial_conditions(self, boundary_conditions):
        self._boundary_conditions = boundary_conditions

    def __getitem__(self, key):
        return self.rhs[key]
