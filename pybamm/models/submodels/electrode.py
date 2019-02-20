#
# Equation classes for the electrode
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Standard(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Current in the
    electrolyte.

    Parameters
    ----------
    phi: :class:`pybamm.Variable`
        A variable representing the electric potential in the electrode
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

    def __init__(self, phi, G):
        super().__init__()

        if phi.domain[0] == "negative electrode":
            sigma = pybamm.standard_parameters.sigma_n
            G = G.orphans[0]
        elif phi.domain[0] == "positive electrode":
            sigma = pybamm.standard_parameters.sigma_p
            G = G.orphans[2]
        else:
            raise pybamm.ModelError("Domain not valid for the electrode equations")

        current = pybamm.Scalar(1)  # maybe change this to the function form later

        i = -sigma * pybamm.grad(phi)

        self.rhs = {}
        self.algebraic = {phi: pybamm.grad(i) - G}
        self.initial_conditions = {}
        if phi.domain[0] == "negative electrode":
            self.boundary_conditions = {i: {"left": current, "right": pybamm.Scalar(0)}}
            self.variables = {"phi_n": phi, "i_n": i}
        elif phi.domain[0] == "positive electrode":
            self.boundary_conditions = {i: {"left": pybamm.Scalar(0), "right": current}}
            self.variables = {"phi_p": phi, "i_p": i}
