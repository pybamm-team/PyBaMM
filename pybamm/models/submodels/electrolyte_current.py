#
# Equation classes for the electrolyte current
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class FirstOrderPotential(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Current in the
    electrolyte.

    Parameters
    ----------
    leading_order_model : Model class
        The leading-order model for the asymptotics
    c_e: :class:`pybamm.Variable`
        A variable representing the concentration of ions in the electrolyte
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, leading_order_model, c_e, param):
        super().__init__()

        # Current function
        i_cell = param.current_with_time

        # Extract leading-order variables, taking orphans to remove domains
        c_e_0 = leading_order_model.variables["Concentration"].orphans[0]
        eps_0 = leading_order_model.variables["Porosity"]
        eps_0n, eps_0s, eps_0p = [e.orphans[0] for e in eps_0.orphans]
        eta_0n = leading_order_model.variables[
            "Negative electrode overpotential"
        ].orphans[0]
        eta_0p = leading_order_model.variables[
            "Positive electrode overpotential"
        ].orphans[0]
        Phi_0 = leading_order_model.variables["Electrolyte potential"].orphans[0]
        V_0 = leading_order_model.variables["Voltage"].orphans[0]

        # Pre-define functions of leading-order variables
        chi_0 = param.chi(c_e_0)
        kappa_0n = param.kappa_e(c_e_0) * eps_0n ** param.b
        kappa_0s = param.kappa_e(c_e_0) * eps_0s ** param.b
        kappa_0p = param.kappa_e(c_e_0) * eps_0p ** param.b
        j0_0n = pybamm.Scalar(1)
        j0_0p = pybamm.Scalar(1)
        j0_1n = pybamm.Scalar(1)
        j0_1p = pybamm.Scalar(1)
        dUPbdc = pybamm.Scalar(1)
        dUPbO2dc = pybamm.Scalar(1)

        # Independent variables
        x_n = pybamm.SpatialVariable("x", ["negative electrode"])
        x_s = pybamm.SpatialVariable("x", ["separator"])
        x_p = pybamm.SpatialVariable("x", ["positive electrode"])

        # First-order concentration (c = c_e_0 + C_e * c_e_1)
        c_e_1 = (c_e - c_e_0) / param.C_e
        c_e_n = c_e.orphans[0]
        c_e_1n = (c_e_n - c_e_0) / param.C_e
        c_e_p = c_e.orphans[2]
        c_e_1p = (c_e_p - c_e_0) / param.C_e

        # Potential
        cbar_1n = pybamm.Scalar(1)  # pybamm.Integral(c_e_1n, x_n) / param.l_n
        j0bar_1n = pybamm.Scalar(1)  # pybamm.Integral(j0_1n, x_n) / param.l_n
        A_n = (
            j0bar_1n * pybamm.Function(np.tanh, eta_0n) / j0_0n
            - dUPbdc * cbar_1n
            - chi_0 / c_e_0 * cbar_1n
            + i_cell * param.l_n / (6 * kappa_0n)
        )

        Phi_1n = -i_cell * x_n ** 2 / (2 * param.l_n * kappa_0n)
        Phi_1s = -i_cell * ((x_s - param.l_n) / kappa_0s + param.l_n / (2 * kappa_0n))
        Phi_1p = -i_cell * (
            param.l_n / (2 * kappa_0n)
            + param.l_s / (kappa_0s)
            + (param.l_p ** 2 - (1 - x_p) ** 2) / (2 * param.l_p * kappa_0p)
        )
        Phi_1 = (
            chi_0 / c_e_0 * c_e_1
            + pybamm.Concatenation(
                pybamm.Broadcast(Phi_1n, ["negative electrode"]),
                pybamm.Broadcast(Phi_1s, ["separator"]),
                pybamm.Broadcast(Phi_1p, ["positive electrode"]),
            )
            + A_n
        )

        # Voltage
        cbar_1p = pybamm.Scalar(1)  # pybamm.Integral(c_e_1p, x_p) / param.l_p
        Phibar_1p = pybamm.Scalar(1)  # pybamm.Integral(Phi1, x_p) / param.l_p
        j0bar_1p = pybamm.Scalar(1)  # pybamm.Integral(j0_1p, x_p) / param.l_p
        V_1 = (
            Phibar_1p
            + dUPbO2dc * cbar_1p
            - j0bar_1p * pybamm.Function(np.tanh, eta_0p) / j0_0p
        )

        # Variables
        self.variables = {
            "c": c_e,
            "Phi": Phi_0 + param.C_e * Phi_1,
            "V": V_0 + param.C_e * V_1,
        }
