#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class ReactionDiffusionModel(pybamm.BaseModel):
    """Reaction-diffusion model.

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
        super()
        self.name = "Reaction Diffusion"
        # Assign tests as an attribute
        if tests:
            assert set(tests.keys()) == {
                "inits",
                "bcs",
                "sources",
            }, "tests.keys() must include, 'inits', 'bcs' and 'sources'"
        self.tests = tests

        # Set variables
        self.variables = [("c", "xc")]

        # Initialise the class(es) that will be called upon for equations
        self.electrolyte = pybamm.Electrolyte()

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

        # Set simulation for the components
        self.electrolyte.set_simulation(param, operators, mesh)

    def initial_conditions(self):
        """See :meth:`pybamm.BaseModel.initial_conditions`"""
        if not self.tests:
            electrolyte_inits = self.electrolyte.initial_conditions()
            y0 = electrolyte_inits["c"]
            return y0
        else:
            return self.tests["inits"]

    def pdes_rhs(self, vars):
        """See :meth:`pybamm.BaseModel.pdes_rhs`"""
        if not self.tests:
            j = np.concatenate(
                [
                    0 * vars.cn + self.param.icell(vars.t) / self.param.ln,
                    0 * vars.cs,
                    0 * vars.cp - self.param.icell(vars.t) / self.param.lp,
                ]
            )
            flux_bcs = self.electrolyte.bcs_cation_flux()
        else:
            flux_bcs = self.tests["bcs"](vars.t)["concentration"]
            # Set s to 1 so that we can provide any source term
            self.param.s = 1
            j = self.tests["sources"](vars.t)["concentration"]
        dcdt = self.electrolyte.cation_conservation(vars.c, j, flux_bcs)

        return dcdt
