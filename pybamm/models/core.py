#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BaseModel(object):
    """Base model class for other models to extend."""

    def __init__(self, tests={}):
        self.name = "Base Model"
        # Assign tests as an attribute
        if tests:
            assert set(tests.keys()) == {
                "inits",
                "bcs",
                "sources",
            }, "tests.keys() must include, 'inits', 'bcs' and 'sources'"
        self.tests = tests

    def __str__(self):
        return self.name

    @property
    def variables(self):
        """The variables of the model, as defined by submodels."""
        return [submodel.variables for submodel in self.submodels]

    @property
    def domains(self):
        """The domain(s) in which the model holds."""
        return set([domain for variable, domain in self.variables])

    def set_simulation(self, param, operators, mesh):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.Parameters` instance
            The parameters of the simulation
        operators : :class:`pybamm.Operators` instance
            The spatial operators.
        mesh : :class:`pybamm.Mesh` instance
            The spatial and temporal discretisation.
        """
        self.param = param
        self.operators = operators
        self.mesh = mesh

    def initial_conditions(self):
        """
        Calculate the initial conditions for the simulation.

        Returns
        -------
        array_like
            A concatenated vector of all the initial conditions.

        """
        return np.concatenate(
            [submodel.initial_conditions(vars) for submodel in self.submodels]
        )

    def pdes_rhs(self, vars):
        """
        Calculate the spatial derivates of the spatial terms in the PDEs and returns
        the right-hand side to be used by the ODE solver (Method of Lines).

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.

        Returns
        -------
        array_like
            A concatenated vector of all the derivatives.

        """
        return np.concatenate([submodel.pdes_rhs(vars) for submodel in self.submodels])

    @property
    def submodels(self):
        raise NotImplementedError
