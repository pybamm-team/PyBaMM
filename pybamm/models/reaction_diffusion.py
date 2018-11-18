#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class ReactionDiffusionModel(pybamm.BaseModel):
    """The (dimensionless) physical/chemical model to be used in the simulation.

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
        self.param = param
        self.operators = operators
        self.mesh = mesh

        # Set simulation for the components
        self.electrolyte.set_simulation(param, operators, mesh)

    def initial_conditions(self):
        """Calculates the initial conditions for the simulation.

        Returns
        -------
        y0 : array_like
            A concatenated vector of all the initial conditions.

        """
        if not self.tests:
            electrolyte_inits = self.electrolyte.initial_conditions()
            c0 = electrolyte_inits["c"]
            return c0
        else:
            return self.tests["inits"]

    def pdes_rhs(self, vars):
        """Calculates the spatial derivates of the spatial terms in the PDEs
           and returns the right-hand side to be used by the ODE solver
           (Method of Lines).

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.

        Returns
        -------
        dydt : array_like
            A concatenated vector of all the derivatives.

        """
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
