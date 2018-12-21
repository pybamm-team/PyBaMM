#
# Equation classes for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class StefanMaxwellDiffusion(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

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

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, G):
        super().__init__()

        electrolyte_domain = [
            "negative electrode",
            "separator",
            "positive " "electrode",
        ]

        c_e = pybamm.Variable("c_e", domain=electrolyte_domain)

        delta = pybamm.Parameter("delta")
        epsilon = pybamm.Parameter("epsilon", domain=electrolyte_domain)
        b = pybamm.Parameter("b")
        D_e = pybamm.Parameter("D_e")
        nu = pybamm.Parameter("nu")
        t_plus = pybamm.Parameter("t_plus")

        c_e0 = pybamm.Parameter("c_e0")  # Should this be a parameter?

        # Change expression once Binary operations can cope with None input
        N_e = pybamm.Scalar(0) - pybamm.Gradient(c_e)

        self.rhs = {
            c_e: pybamm.Scalar(0)
            - pybamm.Divergence(N_e) / delta / epsilon
            + nu * (pybamm.Scalar(1) - t_plus) * G
        }
        self.initial_conditions = {c_e: c_e0}
        self.boundary_conditions = {N_e: {"left": 0, "right": 0}}
        self.variables = {"c_e": c_e, "N_e": N_e}
