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
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        # Initialise empty model
        self._rhs = {}
        self._initial_conditions = {}
        self._boundary_conditions = {}
        self._variables = {}
        self._concatenated_rhs = None
        self._concatenated_initial_conditions = None

        # Default parameter values, discretisation and solver
        self.default_parameter_values = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv"
        )
        mesh = pybamm.FiniteVolumeMacroMesh(self.default_parameter_values, 2)
        self.default_discretisation = pybamm.FiniteVolumeDiscretisation(mesh)
        self.default_solver = pybamm.ScipySolver(method="RK45")

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
        """
        Set initial conditions, converting any scalar conditions to 'pybamm.Scalar'
        and checking that domains are consistent
        """
        # Convert any numbers to a pybamm.Scalar
        for var, eqn in initial_conditions.items():
            if isinstance(eqn, numbers.Number):
                initial_conditions[var] = pybamm.Scalar(eqn)

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

    def update(self, *submodels):
        """
        Update model to add new physics from submodels

        Parameters
        ----------
        submodel : iterable of submodels (subclasses of :class:`pybamm.BaseModel`)
            The submodels from which to create new model
        """
        for submodel in submodels:
            # check for duplicates in keys
            vars = [var.id for var in submodel.rhs.keys()] + [
                var.id for var in self.rhs.keys()
            ]
            assert len(vars) == len(set(vars)), pybamm.ModelError("duplicate variables")

            # update dicts
            self._rhs.update(submodel.rhs)
            self._initial_conditions.update(submodel.initial_conditions)
            self._boundary_conditions.update(submodel.boundary_conditions)
            self._variables.update(submodel.variables)

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
