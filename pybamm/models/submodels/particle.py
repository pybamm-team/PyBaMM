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
    j : :class:`pybamm.Concatenation`
        An expression tree that represents the current density at the
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

    def __init__(self, c, j, param):
        super().__init__()

        if len(c.domain) != 1:
            raise NotImplementedError(
                "Only implemented when c_k is on exactly 1 subdomain"
            )

        if c.domain[0] == "negative particle":
            N = -(1 / param.C_n) * pybamm.grad(c)
            self.rhs = {c: -pybamm.div(N)}
            self.algebraic = {}
            self.initial_conditions = {c: param.c_n_init}
            self.boundary_conditions = {
                N: {"left": pybamm.Scalar(0), "right": j / param.beta_n}
            }
            self.variables = {
                "Negative particle concentration": c,
                "Negative particle surface concentration": pybamm.surf(c),
                "Negative particle flux": N,
            }
        elif c.domain[0] == "positive particle":
            N = -(1 / param.C_p) * pybamm.grad(c)
            self.rhs = {c: -pybamm.div(N)}
            self.algebraic = {}
            self.initial_conditions = {c: param.c_p_init}
            self.boundary_conditions = {
                N: {
                    "left": pybamm.Scalar(0),
                    "right": j / param.beta_p / param.gamma_hat_p,
                }
            }
            self.variables = {"c_s_p": c, "N_s_p": N}
        else:
            raise pybamm.ModelError("Domain not valid for the particle equations")
