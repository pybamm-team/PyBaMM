#
# Parameter functions for the lead-acid models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


def D_hat(c):
    """
    Dimensional effective Fickian diffusivity in the electrolyte [m2.s-1].
    """
    return (1.75 + 260e-6 * c) * 1e-9


def D_eff(param, c, eps):
    """Dimensionless effective Fickian diffusivity in the electrolyte."""
    return D_hat(c * param._raw["cmax"]) / D_hat(param._raw["cmax"]) * (eps ** 1.5)


def kappa_hat(param, c):
    """Dimensional effective conductivity in the electrolyte [S.m-1]"""
    return c * np.exp(6.23 - 1.34e-4 * c - 1.61e-8 * c ** 2) * 1e-4


def kappa_eff(param, c, eps):
    """Dimensionless molar conductivity in the electrolyte"""
    kappa_scale = (
        param._raw["F"] ** 2
        * param._raw["cmax"]
        * D_hat(param._raw["cmax"])
        / (param._raw["R"] * param._raw["T_ref"])
    )
    return kappa_hat(c * param._raw["cmax"]) / kappa_scale * (eps ** 1.5)


def chi_hat(c):
    """Dimensional Darken thermodynamic factor in the electrolyte [-]"""
    return 0.49 + 4.1e-4 * c


def chi(param, c):
    """Dimensionless Darken thermodynamic factor in the electrolyte"""
    # Excluded volume fraction
    alpha = (2 * param._raw["Vw"] - param._raw["Ve"]) * param.scales["conc"]
    chi_scale = 1 / (2 * (1 - param._raw["tpw"]))
    return chi_hat(c * param._raw["cmax"]) / chi_scale / (1 + alpha * c)


# def curlyK_hat(param, eps):
#     """Dimensional permeability [m2]"""
#     return eps ** 3 * param._raw["d"] ** 2 / (180 * (1 - eps) ** 2)
#
#
# def curlyK(param, eps):
#     """Dimensionless permeability"""
#     return curlyK_hat(eps) / param._raw["d"] ** 2
#


def mu_hat(c):
    """Dimensional viscosity of electrolyte [kg.m-1.s-1]"""
    return 0.89e-3 + 1.11e-7 * c + 3.29e-11 * c ** 2


#
# def mu(param, c):
#     """Dimensionless viscosity of electrolyte"""
#     return mu_hat(c * param._raw["cmax"]) / mu_hat(param._raw["cmax"])
#
#
# def rho_hat(param, c):
#     """Dimensional density of electrolyte [kg.m-3]"""
#     return (
#         Mw
#         / param._raw["Vw"]
#         * (
#             1
#             + (
#                 param._raw["Me"] * param._raw["Vw"] / param._raw["Mw"]
#                 - param._raw["Ve"]
#             )
#             * c
#         )
#     )
#
#
# def rho(param, c):
#     """Dimensionless density of electrolyte"""
#     return rho_hat(c * param._raw["cmax"]) / rho_hat(param._raw["cmax"])


def cw_hat(param, c):
    """Dimensional solvent concentration [mol.m-3]"""
    return (1 - c * param._raw["Ve"]) / param._raw["Vw"]


def cw(param, c):
    """Dimensionless solvent concentration"""
    return cw_hat(param, c * param._raw["cmax"]) / cw_hat(param, param._raw["cmax"])


# def dcwdc(param, c):
#     """Dimensionless derivative of cw with respect to c"""
#     return 0 * c - param._raw["Ve"] / param._raw["Vw"]


def m(param, c):
    """Dimensional electrolyte molar mass [mol.kg-1]"""
    return c * param._raw["Vw"] / ((1 - c * param._raw["Ve"]) * param._raw["Mw"])


# def dmdc(param, c):
#     """Dimensional derivative of m with respect to c [kg-1]"""
#     return param.Vw / ((1 - c * param._raw["Ve"]) ** 2 * param._raw["Mw"])


def U_Pb(param, c):
    """Dimensionless OCP in the negative electrode"""
    m_ = m(param, c * param._raw["cmax"])  # dimensionless
    U = (
        param._raw["F"]
        / (param._raw["R"] * param._raw["T_ref"])
        * (
            -0.074 * np.log10(m_)
            - 0.030 * np.log10(m_) ** 2
            - 0.031 * np.log10(m_) ** 3
            - 0.012 * np.log10(m_) ** 4
        )
    )
    return U


# def U_Pb_hat(param, c):
#     """Dimensional OCP in the negative electrode [V]"""
#     return param._raw["U_Pb_ref"] + param._raw["R"] * param._raw["T_ref"] /param._raw[
#         "F"
#     ] * U_Pb(c / param._raw["cmax"])


# def dUPbdc(param, c):
#     """Dimensionless derivative of U_Pb with respect to c"""
#     m_ = m(c * param._raw["cmax"])  # dimensionless
#     dUdm = (
#         param._raw["F"]
#         / (param._raw["R"] * param._raw["T_ref"])
#         * (
#             -0.074 / m_ / np.log(10)
#             - 0.030 * 2 * np.log(m_) / (m_ * np.log(10) ** 2)
#             - 0.031 * 3 * np.log(m_) ** 2 / m_ / np.log(10) ** 3
#             - 0.012 * 4 * np.log(m_) ** 3 / m_ / np.log(10) ** 4
#         )
#     )
#     dmdc = dmdc(c * param._raw["cmax"]) * param._raw["cmax"]  # dimensionless
#     return dmdc * dUdm


def U_PbO2(param, c):
    """Dimensionless OCP in the positive electrode"""
    m_ = m(param, c * param._raw["cmax"])
    U = (
        param._raw["F"]
        / (param._raw["R"] * param._raw["T_ref"])
        * (
            0.074 * np.log10(m_)
            + 0.033 * np.log10(m_) ** 2
            + 0.043 * np.log10(m_) ** 3
            + 0.022 * np.log10(m_) ** 4
        )
    )
    return U


# def U_PbO2_hat(param, c):
#     """Dimensional OCP in the positive electrode [V]"""
#     return param._raw["U_PbO2_ref"] + param._raw["R"] * param._raw[
#         "T_ref"
#     ] / param._raw["F"] * U_PbO2(c / param._raw["cmax"])
#
#
# def dUPbO2dc(param, c):
#     """Dimensionless derivative of U_PbO2 with respect to c"""
#     m_ = m(c * param._raw["cmax"])  # dimensionless
#     dUdm = (
#         param._raw["F"]
#         / (param._raw["R"] * param._raw["T_ref"])
#         * (
#             0.074 / m_ / np.log(10)
#             + 0.033 * 2 * np.log(m_) / (m_ * np.log(10) ** 2)
#             + 0.043 * 3 * np.log(m_) ** 2 / m_ / np.log(10) ** 3
#             + 0.022 * 4 * np.log(m_) ** 3 / m_ / np.log(10) ** 4
#         )
#     )
#     dmdc = dmdc(c * param._raw["cmax"]) * param._raw["cmax"]  # dimensionless
#     return dmdc * dUdm
