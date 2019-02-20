#
# Equation classes for a Particle
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Standard(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Current in the
    electrolyte.

    Parameters
    ----------
    c: :class:`pybamm.Variable`
        A variable representing the lithium concentration in the particle
    G : :class:`pybamm.Concatenation`
        An expression tree that represents the concentration flux at the
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

    def __init__(self, c, G):
        super().__init__()

        if c.domain[0] == "negative particle":

            gamma = pybamm.standard_parameters.gamma_n
            beta = pybamm.standard_parameters.beta_n
            C = pybamm.standard_parameters.C_n
            G = G.orphans[0]
        elif c.domain[0] == "positive particle":
            gamma = pybamm.standard_parameters.gamma_p
            G = G.orphans[2]
        else:
            raise pybamm.ModelError("Domain not valid for the electrode equations")

        N = -gamma * pybamm.grad(c)

        self.rhs = {-pybamm.div(N)}
        self.algebraic = {}
        self.initial_conditions = {}
        self.boundary_conditions = {
            N: {"left": pybamm.Scalar(0), "right": G / beta / C}
        }

        if c.domain[0] == "negative electrode":
            self.variables = {"c_n": c, "N_n": N}
        elif c.domain[0] == "positive electrode":
            self.variables = {"c_p": c, "N_p": N}
