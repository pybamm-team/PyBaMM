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
    G : :class:`pybamm.Symbol`
        An expression tree that represents the cation flux at the electrode-electrolyte
        interface

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

        epsilon = pybamm.standard_parameters.epsilon_s  # make issue for spatially
        # dependent parameters
        b = pybamm.standard_parameters.b
        delta = pybamm.standard_parameters.delta
        nu = pybamm.standard_parameters.nu
        t_plus = pybamm.standard_parameters.t_plus
        ce0 = pybamm.standard_parameters.ce0

        electrolyte_domain = ["whole cell"]

        c_e = pybamm.Variable("c_e", electrolyte_domain)

        N_e = -(epsilon ** b) * pybamm.grad(c_e)

        self.rhs = {c_e: -pybamm.div(N_e) / delta / epsilon + nu * (1 - t_plus) * G}
        self.initial_conditions = {c_e: ce0}
        self.boundary_conditions = {
            N_e: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        self.variables = {"c_e": c_e, "N_e": N_e}
