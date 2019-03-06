#
# Equation classes for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class StefanMaxwell(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Current in the
    electrolyte.

    Parameters
    ----------
    c_e: :class:`pybamm.Variable`
        A variable representing the concentration of ions in the electrolyte
    phi_e: :class:`pybamm.Variable`
        A variable representing the electric potential in the electrolyte
    G : :class:`pybamm.Symbol`
        An expression tree that represents the concentration flux at the
        electrode-electrolyte interface

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, phi_e, G):
        super().__init__()

        epsilon = pybamm.standard_parameters.epsilon_s  # make issue for spatially
        # dependent parameters
        b = pybamm.standard_parameters.b
        delta = pybamm.standard_parameters.delta
        nu = pybamm.standard_parameters.nu
        t_plus = pybamm.standard_parameters.t_plus
        Lambda = pybamm.standard_parameters.Lambda
        sigma = pybamm.Scalar(1)  # leave as a constant for now (should be sigma(c_e))

        i_e = (
            ((epsilon ** b) / delta / nu)
            * sigma
            * (-Lambda * pybamm.grad(phi_e) + 2 * (1 - t_plus) * pybamm.grad(c_e) / c_e)
        )

        self.rhs = {}
        self.algebraic = {phi_e: pybamm.grad(i_e) - G}
        self.initial_conditions = {}
        self.boundary_conditions = {
            i_e: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        self.variables = {"phi_e": phi_e, "i_e": i_e}
