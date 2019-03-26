#
# Standard parameters for lead-acid battery models
#
"""
Standard Parameters for lead-acid battery models, to complement the ones given in
:module:`pybamm.standard_parameters`

Electrolyte Properties
----------------------
ce_typ
    Typical lithium ion concentration in electrolyte
De_typ
    Typical lithium ion diffusivity in the electrolyte
nu_plus
    Stoichiometry of hydrogen anions
nu_minus
    Stoichiometry of hydrogen sulfate anions
nu
    Stoichiometry of sulfuric acid
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

sp = pybamm.standard_parameters

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Electrolyte properties
M_w = pybamm.Parameter("Molar mass of water")
M_p = pybamm.Parameter("Molar mass of cations")
M_n = pybamm.Parameter("Molar mass of anions")
M_e = M_n + M_p  # Molar mass of electrolyte [kg.mol-1]
DeltaVliq_n = (
    sp.V_minus - sp.V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
DeltaVliq_p = (
    2 * sp.V_w - sp.V_minus - 3 * sp.V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

# Electrode properties
sp.V_Pb = pybamm.Parameter("Molar volume of lead")
sp.V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide")
sp.V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate")
DeltaVsurf_n = (
    sp.V_Pb - sp.V_PbSO4
)  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurf_p = (
    sp.V_PbSO4 - sp.V_PbO2
)  # Net Molar Volume consumed in pos electrode [m3.mol-1]
d = pybamm.Parameter("Pore size")
eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
eps_s_max = pybamm.Parameter("Maximum porosity of separator")
eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")

# --------------------------------------------------------------------------------------
"""Functions"""


def chi_dimensional(c_e):
    return pybamm.FunctionParameter("Darken thermodynamic factor", c_e)


# (1-2*sp.t_plus) is for Nernst-Planck
# 2*(1-sp.t_plus) for Stefan-Maxwell
def chi(c_e):
    c_e_dimensional = c_e * sp.c_e_typ
    alpha = (sp.nu * sp.V_w - sp.V_e) * sp.c_e_typ
    return chi_dimensional(c_e_dimensional) * 2 * (1 - sp.t_plus) / (1 - alpha * c_e)


def rho_dimensional(c_e):
    """
    Dimensional density of electrolyte [kg.m-3], from thermodynamics. c_e in [mol.m-3].

    """
    return M_w / sp.V_w * (1 + (M_e * sp.V_w / M_w - sp.V_e) * c_e)


def m_dimensional(c_e):
    """
    Dimensional electrolyte molar mass [mol.kg-1], from thermodynamics.
    c_e in [mol.m-3].

    """
    return c_e * sp.V_w / ((1 - c_e * sp.V_e) * M_w)


def mu_dimensional(c_e):
    """
    Dimensional viscosity of electrolyte [kg.m-1.s-1].

    """
    return pybamm.FunctionParameter("Electrolyte viscosity", c_e)


def U_n_dimensional(c_e):
    "Dimensional open-circuit voltage in the negative electrode [V]"
    return pybamm.FunctionParameter("Negative electrode OCV", m_dimensional(c_e))


U_n_ref = pybamm.FunctionParameter("Negative electrode OCV", pybamm.Scalar(1))


def U_p_dimensional(c_e):
    "Dimensional open-circuit voltage in the positive electrode [V]"
    return pybamm.FunctionParameter("Positive electrode OCV", m_dimensional(c_e))


U_p_ref = pybamm.FunctionParameter("Positive electrode OCV", pybamm.Scalar(1))


def U_n(c_en):
    "Dimensionless open-circuit voltage in the negative electrode"
    c_en_dimensional = c_en * sp.c_e_typ
    return (U_n_dimensional(c_en_dimensional) - U_n_ref) / sp.potential_scale


def U_p(c_ep):
    "Dimensionless open-circuit voltage in the positive electrode"
    c_ep_dimensional = c_ep * sp.c_e_typ
    return (U_p_dimensional(c_ep_dimensional) - U_p_ref) / sp.potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""

tau_discharge = sp.F * sp.c_e_typ * sp.L_x / sp.i_typ
velocity_scale = sp.i_typ / (sp.c_e_typ * sp.F)  # Reaction velocity scale

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Electrolyte properties
beta_surf_n = -sp.c_e_typ * DeltaVsurf_n / sp.ne_n  # Molar volume change (lead)
beta_surf_p = -sp.c_e_typ * DeltaVsurf_p / sp.ne_p  # Molar volume change (lead dioxide)
beta_surf = pybamm.Concatenation(
    pybamm.Broadcast(beta_surf_n, ["negative electrode"]),
    pybamm.Broadcast(0, ["separator"]),
    pybamm.Broadcast(beta_surf_p, ["positive electrode"]),
)
beta_liq_n = (
    -sp.c_e_typ * DeltaVliq_n / sp.ne_n
)  # Molar volume change (electrolyte, neg)
beta_liq_p = (
    -sp.c_e_typ * DeltaVliq_p / sp.ne_p
)  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    sp.c_e_typ * M_e / rho_dimensional(sp.c_e_typ) * (1 - M_w * sp.V_e / sp.V_w * M_e)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    sp.c_e_typ * M_e / rho_dimensional(sp.c_e_typ) * (sp.t_plus + M_n / M_e)
)  # Migrative kinematic relationship coefficient
C_e = sp.tau_diffusion_e / tau_discharge
pi_os = (
    mu_dimensional(sp.c_e_typ)
    * velocity_scale
    * sp.L_x
    / (d ** 2 * sp.R * sp.T * sp.c_e_typ)
)  # Ratio of viscous pressure scale to osmotic pressure scale
gamma_hat_e = 1  # ratio of electrolyte concentration to electrode concentration, undef.

# Electrochemical reactions
C_dl_n = (
    sp.C_dl_dimensional
    * sp.potential_scale
    / sp.interfacial_current_scale_n
    / tau_discharge
)
C_dl_p = (
    sp.C_dl_dimensional
    * sp.potential_scale
    / sp.interfacial_current_scale_p
    / tau_discharge
)

# Electrical
voltage_low_cut = (
    sp.voltage_low_cut_dimensional - (U_p_ref - U_n_ref)
) / sp.potential_scale
voltage_high_cut = (
    sp.voltage_high_cut_dimensional - (U_p_ref - U_n_ref)
) / sp.potential_scale

# Initial conditions
q_init = pybamm.Parameter("Initial State of Charge")
q_max = (
    (sp.L_n * eps_n_max + sp.L_s * eps_s_max + sp.L_p * eps_p_max)
    / sp.L_x
    / (sp.s_p - sp.s_n)
)  # Dimensionless max capacity
epsDelta_n = beta_surf_n / sp.L_n * q_max
epsDelta_p = beta_surf_p / sp.L_p * q_max
c_e_init = q_init
eps_n_init = eps_n_max - epsDelta_n * (1 - q_init)  # Initial pororsity (neg) [-]
eps_s_init = eps_s_max  # Initial pororsity (sep) [-]
eps_p_init = eps_p_max - epsDelta_p * (1 - q_init)  # Initial pororsity (pos) [-]
eps_init = pybamm.Concatenation(
    pybamm.Broadcast(eps_n_init, ["negative electrode"]),
    pybamm.Broadcast(eps_s_init, ["separator"]),
    pybamm.Broadcast(eps_p_init, ["positive electrode"]),
)
