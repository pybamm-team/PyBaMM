#
# Electrolyte current model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class ElectrolyteCurrentModel(pybamm.BaseModel):
    """Electrolyte current model.

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
        self.name = "Electrolyte Current"
        # Assign tests as an attribute
        if tests:
            assert set(tests.keys()) == {
                "inits",
                "bcs",
                "sources",
            }, "tests.keys() must include, 'inits', 'bcs' and 'sources'"
        self.tests = tests

        # Set variables
        self.variables = [("en", "xcn"), ("ep", "xcp")]

        # Initialise the class(es) that will be called upon for equations
        self.electrolyte = pybamm.Electrolyte()
        self.interface = pybamm.Interface()

    def set_simulation(self, param, operators, mesh):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.ParameterValues` instance
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
        self.interface.set_simulation(param, mesh)

    def initial_conditions(self):
        """See :meth:`pybamm.BaseModel.initial_conditions`"""
        if not self.tests:
            electrolyte_inits = self.electrolyte.initial_conditions()
            y0 = np.concatenate([electrolyte_inits["en"], electrolyte_inits["ep"]])
            return y0
        else:
            return self.tests["inits"]

    def pdes_rhs(self, vars):
        """See :meth:`pybamm.BaseModel.pdes_rhs`"""
        cn = np.ones_like(self.mesh.xcn)
        cp = np.ones_like(self.mesh.xcp)
        if not self.tests:
            current_bcs_n = self.electrolyte.bcs_current("xcn", vars.t)
            current_bcs_p = self.electrolyte.bcs_current("xcp", vars.t)
            jn = self.interface.butler_volmer("xcn", cn, vars.en)
            jp = self.interface.butler_volmer("xcp", cp, vars.ep)
        else:
            current_bcs_n = self.tests["bcs"](vars.t)["current n"]
            current_bcs_p = self.tests["bcs"](vars.t)["current p"]
            jn = self.tests["sources"](vars.t)["current n"]
            jp = self.tests["sources"](vars.t)["current p"]
        dendt = self.electrolyte.current_conservation(
            "xcn", cn, vars.en, jn, current_bcs_n
        )
        depdt = self.electrolyte.current_conservation(
            "xcp", cp, vars.ep, jp, current_bcs_p
        )

        return np.concatenate([dendt, depdt])
