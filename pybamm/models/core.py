#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class BaseModel(object):
    """Base model class for other models to extend.

    Parameters
    ----------
    tests : dict, optional
        A dictionary for testing the convergence of the numerical solution:
            * {} (default): We are not running in test mode, use built-ins.
            * {'inits': dict of initial conditions,
               'bcs': dict of boundary conditions,
               'sources': dict of source terms
               }: To be used for testing convergence to an exact solution.
    """

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
    def pde_variables(self):
        """The variables of the model, as defined by submodels."""
        return [
            (variable, submodel.submesh)
            for variable, submodel in self.submodels["pdes"].items()
        ]

    # @property
    # def domains(self):
    #     """The domain(s) in which the model holds."""
    #     return set([domain for self.domain in self.variables])

    def set_simulation(self, param, operators, mesh):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.BaseParameterValues` instance
            The parameters of the simulation
        operators : :class:`pybamm.Operators` instance
            The spatial operators.
        mesh : :class:`pybamm.Mesh` instance
            The spatial and temporal discretisation.
        """
        self.param = param
        self.operators = operators
        self.mesh = mesh

        self.simulation_set = True

    def initial_conditions(self):
        """
        Calculate the initial conditions for the simulation.

        Returns
        -------
        array_like
            A concatenated vector of all the initial conditions.

        """
        return np.concatenate(
            [
                submodel.initial_conditions()
                for submodel in self.submodels["pdes"].values()
            ]
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
        dydt : array_like
            A concatenated vector of all the derivatives.

        """
        j = self.reactions(vars)
        vars.set_reaction_vars({"j": j})
        dydt = np.concatenate(
            [submodel.pdes_rhs(vars) for submodel in self.submodels["pdes"].values()]
        )

        return dydt

    def reactions(self, vars):
        """
        Calculate the interfacial current density using the reactions defined in
        self.submodels.

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.

        Returns
        -------
        array_like
            The interfacial current density.

        """
        jn = self.submodels["reactions"]["neg"].reaction(vars.neg)
        jp = self.submodels["reactions"]["pos"].reaction(vars.pos)

        return np.concatenate([jn, np.zeros_like(self.mesh.xs.nodes), jp])

    @property
    def submodels(self):
        raise NotImplementedError
