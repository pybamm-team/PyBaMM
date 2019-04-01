#
# Equation classes for the electrolyte current
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class MacInnesStefanMaxwell(pybamm.LeadAcidBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        The electrolyte concentration
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (can be Variable or Parameter)
    phi_e : :class:`pybamm.Symbol`
        The electric potential in the electrolyte ("electrolyte potential")
    j : :class:`pybamm.Symbol`
        The interfacial current density at the electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, eps, phi_e, j, param):
        super().__init__()

        # functions

        i_e = (
            param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_hat_e
        ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e))

        # Equations (algebraic only)
        self.algebraic = {phi_e: pybamm.div(i_e) - j}
        self.boundary_conditions = {i_e: {"left": 0, "right": 0}}
        self.initial_conditions = {phi_e: param.U_n(param.c_e_init)}
        # no differential equations
        self.rhs = {}
        # Variables
        self.variables = {"Electrolyte potential": phi_e, "Electrolyte current": i_e}

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()


class MacInnesStefanMaxwellCapacitance(pybamm.LeadAcidBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations, with capacitance effects included. The MacInnes equation
    is rearranged to account for both solid and electrolyte potentials

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        The electrolyte concentration
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (can be Variable or Parameter)
    Delta_phi : :class:`pybamm.Symbol`
        The difference between the electric potential in the electrolyte and the
        electric potential in the solid (Delta_phi = phi_s - phi_e)
    j : :class:`pybamm.Symbol`
        The interfacial current density at the electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, eps, Delta_phi, j, param):
        super().__init__()
        current = pybamm.standard_parameters.current_with_time

        # ode model only
        self.algebraic = {}
        i_e = (
            param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_hat_e
        ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e + pybamm.grad(Delta_phi))

        # different bounday conditions in each electrode
        if Delta_phi.domain == ["negative electrode"]:
            self.rhs = {Delta_phi: 1 / param.gamma_dl_n * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {Delta_phi: {"left": 0}, i_s_n: {"right": 0}}
            self.boundary_conditions = {i_e: {"left": 0, "right": current}}
            self.initial_conditions = {Delta_phi: param.U_n(param.c_e_init)}
            self.variables = {
                "Negative electrode potential difference": Delta_phi,
                "Negative electrode electrolyte current": i_e,
            }
        elif Delta_phi.domain == ["positive electrode"]:
            self.rhs = {Delta_phi: 1 / param.gamma_dl_p * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {i_e: {"left": current, "right": 0}}
            self.initial_conditions = {Delta_phi: param.U_p(param.c_e_init)}
            self.variables = {
                "Positive electrode potential difference": Delta_phi,
                "Positive electrode electrolyte current": i_e,
            }
        # for whole cell domain call both electrode models and ignore separator
        elif Delta_phi.domain == [
            "negative electrode",
            "separator",
            "positive electrode",
        ]:
            c_e_n, c_e_s, c_e_p = c_e.orphans
            Delta_phi_n, Delta_phi_s, Delta_phi_p = Delta_phi.orphans
            eps_n, eps_s, eps_p = eps.orphans
            j_n, j_s, j_p = j.orphans
            neg_model = MacInnesStefanMaxwellCapacitance(
                c_e_n, Delta_phi_n, eps_n, j_n, param
            )
            pos_model = MacInnesStefanMaxwellCapacitance(
                c_e_p, Delta_phi_p, eps_p, j_p, param
            )
            self.update(neg_model, pos_model)
            # Voltage variable
            voltage = pybamm.BoundaryValue(phi_s, "right") - pybamm.BoundaryValue(
                phi_s, "left"
            )
            self.variables.update({"Voltage": voltage})
        else:
            raise pybamm.DomainError(
                "domain '{}' not recognised".format(Delta_phi.domain)
            )


class StefanMaxwellFirstOrderPotential(pybamm.BaseModel):
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
        c_e_0 = leading_order_model.variables["Electrolyte concentration"].orphans[0]
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

        # Pre-define functions of leading-order variables
        chi_0 = param.chi(c_e_0)
        kappa_0n = param.kappa_e(c_e_0) * eps_0n ** param.b
        kappa_0s = param.kappa_e(c_e_0) * eps_0s ** param.b
        kappa_0p = param.kappa_e(c_e_0) * eps_0p ** param.b
        j0_0n = pybamm.interface.exchange_current_density(
            c_e_0, domain=["negative electrode"]
        )
        j0_0p = pybamm.interface.exchange_current_density(
            c_e_0, domain=["positive electrode"]
        )
        U_0n = param.U_n(c_e_0)
        U_0p = param.U_p(c_e_0)
        j0_1n = c_e_1n * j0_0n.diff(c_e_0)
        j0_1p = c_e_1p * j0_0p.diff(c_e_0)
        dU_0n__dc0 = U_0n.diff(c_e_0)
        dU_0p__dc0 = U_0p.diff(c_e_0)

        # Potential
        cbar_1n = pybamm.Integral(c_e_1n, x_n) / param.l_n
        j0bar_1n = pybamm.Integral(j0_1n, x_n) / param.l_n
        A_n = (
            j0bar_1n * pybamm.Function(np.tanh, eta_0n) / j0_0n
            - dU_0n__dc0 * cbar_1n
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
        cbar_1p = pybamm.Integral(c_e_1p, x_p) / param.l_p
        Phibar_1p = pybamm.Integral(Phi_1p, x_p) / param.l_p
        j0bar_1p = pybamm.Integral(j0_1p, x_p) / param.l_p
        V_1 = (
            Phibar_1p
            + dU_0p__dc0 * cbar_1p
            - j0bar_1p * pybamm.Function(np.tanh, eta_0p) / j0_0p
        )

        # Variables
        self.variables = {
            "Electrolyte potential": Phi_0 + param.C_e * Phi_1,
            "Voltage": V_0 + param.C_e * V_1,
        }
