#
# Equation classes for the electrolyte porosity
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Standard(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity
    j : :class:`pybamm.Symbol`
        The interfacial current density at the electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, epsilon, j, param):
        super().__init__()

        self.rhs = {epsilon: -param.beta_surf * j}
        self.initial_conditions = {epsilon: param.eps_init}
        self.initial_conditions_ydot = {epsilon: 0}
