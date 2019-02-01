#
# Base model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numbers


class BaseModel(object):
    """Base model class for other models to extend.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    algebraic: dict
        A list of algebraic expressions that are assumed to equate to zero
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions for the state variables y
    initial_conditions_ydot: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions for the time derivative of y
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    """

    def __init__(self):
        # Initialise empty model
        self._rhs = {}
        self._algebraic = []
        self._initial_conditions = {}
        self._initial_conditions_ydot = {}
        self._boundary_conditions = {}
        self._variables = {}
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
    def algebraic(self):
        return self._algebraic

    @algebraic.setter
    def algebraic(self, algebraic):
        self._algebraic = algebraic

    def _set_initial_conditions(self, initial_conditions):
        """
        converte any scalar conditions to 'pybamm.Scalar'
        and checking that domains are consistent
        """
        # Convert any numbers to a pybamm.Scalar
        for var, eqn in initial_conditions.items():
            if isinstance(eqn, numbers.Number):
                initial_conditions[var] = pybamm.Scalar(eqn)

        if not all(
            [
                variable.domain == equation.domain or equation.domain == []
                for variable, equation in initial_conditions.items()
            ]
        ):
            raise pybamm.DomainError(
                """variable and equation in initial_conditions
                   must have the same domain"""
            )

        return initial_conditions

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        self._initial_conditions = self._set_initial_conditions(
            initial_conditions
        )

    @property
    def initial_conditions_ydot(self):
        return self._initial_conditions_ydot

    @initial_conditions_ydot.setter
    def initial_conditions_ydot(self, initial_conditions):
        self._initial_conditions_ydot = self._set_initial_conditions(
            initial_conditions
        )

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        # Convert any numbers to a pybamm.Scalar
        for var, bcs in boundary_conditions.items():
            for side, eqn in bcs.items():
                if isinstance(eqn, numbers.Number):
                    boundary_conditions[var][side] = pybamm.Scalar(eqn)

        self._boundary_conditions = boundary_conditions

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        self._variables = variables

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

    def check_well_posedness(self):
        """
        Check that the model is well-posed by executing the following tests:
        - There is an initial condition in self.initial_conditions for each
        variable/equation pair in self.rhs
        - There are appropriate boundary conditions in self.boundary_conditions for each
        variable/equation pair in self.rhs
        """
        # Initial conditions
        for var in self.rhs.keys():
            assert var in self.initial_conditions.keys(), pybamm.ModelError(
                """no initial condition given for variable '{}'""".format(var)
            )

        # Boundary conditions
        for var, eqn in self.rhs.items():
            if eqn.has_spatial_derivatives():
                # Variable must be in at least one expression in the boundary condition
                # keys (to account for both Dirichlet and Neumann boundary conditions)
                assert any(
                    [
                        any([var.id == symbol.id for symbol in key.pre_order()])
                        for key in self.boundary_conditions.keys()
                    ]
                ), pybamm.ModelError(
                    """
                    no boundary condition given for variable '{}'
                    with equation '{}'
                    """.format(
                        var, eqn
                    )
                )
