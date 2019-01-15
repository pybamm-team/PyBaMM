#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BaseModel(object):
    """Base model class for other models to extend.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    """

    def __init__(self):
        self._rhs = {}
        self._initial_conditions = {}
        self._boundary_conditions = {}
        self._concatenated_rhs = None
        self._concatenated_initial_conditions = None

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, rhs):
        if all(
            [
                variable.domain == equation.domain or equation.domain == []
                for variable, equation in rhs.items()
            ]
        ):
            self._rhs = rhs
        else:
            raise pybamm.DomainError(
                """variable and equation in rhs must have the same domain"""
            )

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        if all(
            [
                variable.domain == equation.domain or equation.domain == []
                for variable, equation in initial_conditions.items()
            ]
        ):
            self._initial_conditions = initial_conditions
        else:
            raise pybamm.DomainError(
                """variable and equation in initial_conditions
                   must have the same domain"""
            )

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        self._boundary_conditions = boundary_conditions

    @property
    def concatenated_rhs(self):
        return self._concatenated_rhs

    @concatenated_rhs.setter
    def concatenated_rhs(self, concatenated_rhs):
        self._concatenated_rhs = concatenated_rhs

    @property
    def concatenated_initial_conditions(self):
        return self._concatenated_initial_conditions

    @concatenated_initial_conditions.setter
    def concatenated_initial_conditions(self, concatenated_initial_conditions):
        self._concatenated_initial_conditions = concatenated_initial_conditions

    def __getitem__(self, key):
        return self.rhs[key]
