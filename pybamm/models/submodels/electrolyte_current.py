#
# Equation classes for the electrolyte current
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class MacInnesStefanMaxwell(pybamm.BaseModel):
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
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, phi_e, j, param, eps=None):
        super().__init__()

        # if porosity is not a variable, use the input parameter
        if eps is None:
            eps = param.epsilon

        # functions
        i_e = (param.kappa_e(c_e) * (eps ** param.b) * param.gamma_e / param.C_e) * (
            param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )

        # Equations (algebraic only)
        self.algebraic = {phi_e: pybamm.div(i_e) - j}
        self.boundary_conditions = {i_e: {"left": 0, "right": 0}}
        self.initial_conditions = {phi_e: -param.U_n(param.c_n_init)}
        # no differential equations
        self.rhs = {}
        # Variables
        self.variables = {"Electrolyte potential": phi_e, "Electrolyte current": i_e}

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()


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


def explicit_combined_stefan_maxwell(param, c_e, ocp_n, eta_r_n, c_e_0=1, eps=None):
    """
    Provides and explicit combined leading and first order solution to the electrolyte
    current conervation equation where the constitutive equation is taken to be of
    Stefan-Maxwell form. Note that the returned current density is only the leading
    order approximation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    c_e : :class:`pybamm.Concatenation`
        The electrolyte concentration (combined leading and first order)
    ocp_n : :class:`pybamm.Symbol`
        Open circuit potential at the point of the cell
    eta_r_n : :class: `pybamm.Symbol`
        Reaction overpotential at the point of the cell
        (combined leading and first order)
    c_e_0 : :class: `pybamm.Symbol`
        Leading-order electrolyte concentration (=1 for lithium-ion)
    eps: :class: `pybamm.Symbol`
        Electrode porosity. If not supplied, porosity values in param are used

    Returns
    -------
    phi_e :class: `pybamm.Concatenation`
        The electrolyte potential (combined leading and first order)
    i_e :class: `pybamm.Concatenation`
        The electrolyte current (leading order)
    Delta_Phi_e: `pybamm.Symbol`
        Average Ohmic losses in the electrolyte (combined leading and first order)
    eta_c: `Pybamm.Symbol`
        Average Concentration overpotential (combined leading and first order)
    """

    # import standard spatial vairables
    x_n = pybamm.standard_spatial_vars.x_n
    x_s = pybamm.standard_spatial_vars.x_s
    x_p = pybamm.standard_spatial_vars.x_p

    # import geometric parameters
    l_n = pybamm.geometric_parameters.l_n
    l_p = pybamm.geometric_parameters.l_p

    # import current
    i_cell = param.current_with_time

    # extract c_e components
    c_e_n, c_e_s, c_e_p = [c for c in c_e.orphans]

    # if porosity is not passed in then use the parameter value
    if eps is None:
        eps = param.epsilon
    eps_n, eps_s, eps_p = [e.orphans[0] for e in eps.orphans]

    # bulk conductivities (leading order)
    kappa_n = param.kappa_e(c_e_0) * eps_n ** param.b
    kappa_s = param.kappa_e(c_e_0) * eps_s ** param.b
    kappa_p = param.kappa_e(c_e_0) * eps_p ** param.b

    # get left-most ocp and overpotential
    ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
    eta_r_n_left = pybamm.BoundaryValue(eta_r_n, "left")
    c_e_n_left = pybamm.BoundaryValue(c_e_n, "left")

    # get explicit leading order current
    _, i_e, _, _ = pybamm.electrolyte_current.explicit_leading_order_stefan_maxwell(
        param, c_e, ocp_n, eta_r_n, eps=eps
    )

    # electrolyte potential (combined leading and first order)
    phi_e_const = (
        -ocp_n_left
        - eta_r_n_left
        - 2
        * param.C_e
        * (1 - param.t_plus)
        * pybamm.Function(np.log, c_e_n_left / c_e_0)
        + param.C_e * i_cell / param.gamma_e * (-l_n / (2 * kappa_n) + (l_n / kappa_s))
    )

    phi_e_n = phi_e_const + param.C_e * (
        +2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_n / c_e_0)
        - (i_cell / param.gamma_e)
        * ((x_n ** 2 - l_n ** 2) / (2 * kappa_n * l_n) + l_n / kappa_s)
    )
    phi_e_s = phi_e_const + param.C_e * (
        +2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_s / c_e_0)
        - (i_cell / param.gamma_e) * (x_s / kappa_s)
    )
    phi_e_p = phi_e_const + param.C_e * (
        +2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_p / c_e_0)
        - (i_cell / param.gamma_e)
        * ((x_p * (2 - x_p) - l_p ** 2 - 1) / (2 * kappa_p * l_p) + (1 - l_p) / kappa_s)
    )

    phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

    "Ohmic losses and overpotentials"
    # average electrolyte ohmic losses
    Delta_Phi_e_av = -(param.C_e * i_cell / param.gamma_e / param.kappa_e(1)) * (
        param.l_n / (3 * param.epsilon_n ** param.b)
        + param.l_s / (param.epsilon_s ** param.b)
        + param.l_p / (3 * param.epsilon_p ** param.b)
    )

    # electrode-averaged electrolye concentrations (combined leading
    # and first order)
    c_e_n_av = pybamm.Integral(c_e_n, x_n) / l_n
    c_e_p_av = pybamm.Integral(c_e_p, x_p) / l_p

    # concentration overpotential (combined leading and first order)
    eta_c_av = 2 * param.C_e * (1 - param.t_plus) * (c_e_p_av - c_e_n_av)

    return phi_e, i_e, Delta_Phi_e_av, eta_c_av


def explicit_leading_order_stefan_maxwell(param, c_e, ocp_n, eta_r_n, eps=None):
    """
    Provides and explicit combined leading and first order solution to the electrolyte
    current conervation equation where the constitutive equation is taken to be of
    Stefan-Maxwell form. Note that the returned current density is only the leading
    order approximation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    c_e : :class:`pybamm.Concatenation`
        The electrolyte concentration (combined leading and first order)
    ocp_n : :class:`pybamm.Symbol`
        Open circuit potential at the point of the cell
    eta_r_n : :class: `pybamm.Symbol`
        Reaction overpotential at the point of the cell
        (combined leading and first order)
    c_e_0 : :class: `pybamm.Symbol`
        Leading-order electrolyte concentration (=1 for lithium-ion)
    eps: :class: `pybamm.Symbol`
        Electrode porosity. If not supplied, porosity values in param are used

    Returns
    -------
    phi_e :class: `pybamm.Concatenation`
        The electrolyte potential (combined leading and first order)
    i_e :class: `pybamm.Concatenation`
        The electrolyte current (leading order)
    Delta_Phi_e: `pybamm.Symbol`
        Average Ohmic losses in the electrolyte (combined leading and first order)
    eta_c: `Pybamm.Symbol`
        Average Concentration overpotential (combined leading and first order)
    """
    # import standard spatial vairables
    x_n = pybamm.standard_spatial_vars.x_n
    x_p = pybamm.standard_spatial_vars.x_p

    # import geometric parameters
    l_n = pybamm.geometric_parameters.l_n
    l_p = pybamm.geometric_parameters.l_p

    # import current
    i_cell = param.current_with_time

    # get left-most ocp and overpotential
    ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
    eta_r_n_left = pybamm.BoundaryValue(eta_r_n, "left")

    # electrolye potential
    phi_e_n = -ocp_n_left - eta_r_n_left + pybamm.Broadcast(0, ["negative electrode"])
    phi_e_s = -ocp_n_left - eta_r_n_left + pybamm.Broadcast(0, ["separator"])
    phi_e_p = -ocp_n_left - eta_r_n_left + pybamm.Broadcast(0, ["positive electrode"])
    phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

    # electrolyte current
    i_e_n = i_cell * x_n / l_n
    i_e_s = pybamm.Broadcast(i_cell, ["separator"])
    i_e_p = i_cell * (1 - x_p) / l_p
    i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

    # electrolyte ohmic losses
    Delta_Phi_e_av = pybamm.Scalar(0)

    # concentration overpotential
    eta_c_av = pybamm.Scalar(0)

    return phi_e, i_e, Delta_Phi_e_av, eta_c_av
