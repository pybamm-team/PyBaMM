#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class BaseModel(object):
    """Base model class for other models to extend.

    Attributes
    ----------

    _rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    _initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    _boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
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
    def boundary_conditions(self, boundary_conditions):
        self._boundary_conditions = boundary_conditions

    def __getitem__(self, key):
        return self.rhs[key]
