#
# Equation classes for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class StefanMaxwellDiffusion(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface

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

    def __init__(self, j):
        super().__init__()

        # Parameters
        sp = pybamm.standard_parameters
        spli = pybamm.standard_parameters_lithium_ion

        electrolyte_domain = ["negative electrode", "separator", "positive electrode"]

        c_e = pybamm.Variable("c_e", electrolyte_domain)

        N_e = -sp.D_e(c_e) * (spli.epsilon ** sp.b) * pybamm.grad(c_e)

        self.rhs = {
            c_e: -pybamm.div(N_e) / spli.C_e / spli.epsilon
            + sp.s / spli.gamma_hat_e * j
        }
        self.initial_conditions = {c_e: spli.c_e_init}
        self.boundary_conditions = {
            N_e: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        self.variables = {"c_e": c_e, "N_e": N_e}
