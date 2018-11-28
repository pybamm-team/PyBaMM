#
# Parameter functions for the lead-acid models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


def D_hat(param, c):
    """
    Dimensional effective Fickian diffusivity in the electrolyte [m2.s-1].
    """
    return (1.75 + 260e-6 * c) * 1e-9


def D_eff(param, c, eps):
    """Dimensionless effective Fickian diffusivity in the electrolyte."""
    return self.D_hat(c * self.cmax) / self.D_hat(self.cmax) * (eps ** 1.5)


def DO2_hat(param, cO2):
    """
    Dimensional effective Fickian diffusivity of oxygen
    in the electrolyte [m2.s-1].
    """
    return 1e-9 * np.ones_like(cO2)


def DO2_eff(param, cO2, eps):
    """
    Dimensionless effective Fickian diffusivity of oxygen
    in the electrolyte.
    """
    return self.DO2_hat(cO2 * self.cO2ref) / self.DO2_hat(self.cO2ref) * (eps ** 1.5)


def kappa_hat(param, c):
    """Dimensional effective conductivity in the electrolyte [S.m-1]"""
    return c * np.exp(6.23 - 1.34e-4 * c - 1.61e-8 * c ** 2) * 1e-4


def kappa_eff(param, c, eps):
    """Dimensionless molar conductivity in the electrolyte"""
    kappa_scale = (
        self.F ** 2 * self.cmax * self.D_hat(self.cmax) / (self.R * self.T_ref)
    )
    return self.kappa_hat(c * self.cmax) / kappa_scale * (eps ** 1.5)


def chi_hat(param, c):
    """Dimensional Darken thermodynamic factor in the electrolyte [-]"""
    return 0.49 + 4.1e-4 * c


def chi(param, c):
    """Dimensionless Darken thermodynamic factor in the electrolyte"""
    chi_scale = 1 / (2 * (1 - self.tpw))
    return self.chi_hat(c * self.cmax) / chi_scale / (1 + self.alpha * c)


def curlyK_hat(param, eps):
    """Dimensional permeability [m2]"""
    return eps ** 3 * self.d ** 2 / (180 * (1 - eps) ** 2)


def curlyK(param, eps):
    """Dimensionless permeability"""
    return self.curlyK_hat(eps) / self.d ** 2


def mu_hat(param, c):
    """Dimensional viscosity of electrolyte [kg.m-1.s-1]"""
    return 0.89e-3 + 1.11e-7 * c + 3.29e-11 * c ** 2


def mu(param, c):
    """Dimensionless viscosity of electrolyte"""
    return self.mu_hat(c * self.cmax) / self.mu_hat(self.cmax)


def rho_hat(param, c):
    """Dimensional density of electrolyte [kg.m-3]"""
    return self.Mw / self.Vw * (1 + (self.Me * self.Vw / self.Mw - self.Ve) * c)


def rho(param, c):
    """Dimensionless density of electrolyte"""
    return self.rho_hat(c * self.cmax) / self.rho_hat(self.cmax)


def cw_hat(param, c):
    """Dimensional solvent concentration [mol.m-3]"""
    return (1 - c * self.Ve) / self.Vw


def cw(param, c):
    """Dimensionless solvent concentration"""
    return self.cw_hat(c * self.cmax) / self.cw_hat(self.cmax)


def dcwdc(param, c):
    """Dimensionless derivative of cw with respect to c"""
    return 0 * c - self.Ve / self.Vw


def m(param, c):
    """Dimensional electrolyte molar mass [mol.kg-1]"""
    return c * self.Vw / ((1 - c * self.Ve) * self.Mw)


def dmdc(param, c):
    """Dimensional derivative of m with respect to c [kg-1]"""
    return self.Vw / ((1 - c * self.Ve) ** 2 * self.Mw)


def U_Pb(param, c):
    """Dimensionless OCP in the negative electrode"""
    m = self.m(c * self.cmax)  # dimensionless
    U = (
        self.F
        / (self.R * self.T_ref)
        * (
            -0.074 * np.log10(m)
            - 0.030 * np.log10(m) ** 2
            - 0.031 * np.log10(m) ** 3
            - 0.012 * np.log10(m) ** 4
        )
    )
    return U


def U_Pb_hat(param, c):
    """Dimensional OCP in the negative electrode [V]"""
    return self.U_Pb_ref + self.R * self.T_ref / self.F * self.U_Pb(c / self.cmax)


def dUPbdc(param, c):
    """Dimensionless derivative of U_Pb with respect to c"""
    m = self.m(c * self.cmax)  # dimensionless
    dUdm = (
        self.F
        / (self.R * self.T_ref)
        * (
            -0.074 / m / np.log(10)
            - 0.030 * 2 * np.log(m) / (m * np.log(10) ** 2)
            - 0.031 * 3 * np.log(m) ** 2 / m / np.log(10) ** 3
            - 0.012 * 4 * np.log(m) ** 3 / m / np.log(10) ** 4
        )
    )
    dmdc = self.dmdc(c * self.cmax) * self.cmax  # dimensionless
    return dmdc * dUdm


def U_PbO2(param, c):
    """Dimensionless OCP in the positive electrode"""
    m = self.m(c * self.cmax)
    U = (
        self.F
        / (self.R * self.T_ref)
        * (
            0.074 * np.log10(m)
            + 0.033 * np.log10(m) ** 2
            + 0.043 * np.log10(m) ** 3
            + 0.022 * np.log10(m) ** 4
        )
    )
    return U


def U_PbO2_hat(param, c):
    """Dimensional OCP in the positive electrode [V]"""
    return self.U_PbO2_ref + self.R * self.T_ref / self.F * self.U_PbO2(c / self.cmax)


def dUPbO2dc(param, c):
    """Dimensionless derivative of U_PbO2 with respect to c"""
    m = self.m(c * self.cmax)  # dimensionless
    dUdm = (
        self.F
        / (self.R * self.T_ref)
        * (
            0.074 / m / np.log(10)
            + 0.033 * 2 * np.log(m) / (m * np.log(10) ** 2)
            + 0.043 * 3 * np.log(m) ** 2 / m / np.log(10) ** 3
            + 0.022 * 4 * np.log(m) ** 3 / m / np.log(10) ** 4
        )
    )
    dmdc = self.dmdc(c * self.cmax) * self.cmax  # dimensionless
    return dmdc * dUdm
