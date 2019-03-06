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
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, epsilon, j):
        super().__init__()
        sp = pybamm.standard_parameters_lead_acid

        # Model
        self.rhs = {epsilon: -sp.beta_surf * j}
        self.initial_conditions = {epsilon: sp.eps_init}
