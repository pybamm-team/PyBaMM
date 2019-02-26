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
from pybamm.standard_parameters import (
    c_e_typ,
    C_dl,
    F,
    i_typ,
    L_n,
    L_s,
    L_p,
    L_x,
    ne_n,
    ne_p,
    R,
    s_n,
    s_p,
    T,
    t_plus,
    U_n_dimensional,
    U_n_ref,
    U_p_dimensional,
    U_p_ref,
    V_e,
    V_minus,
    V_plus,
    V_w,
    interfacial_current_scale_n,
    interfacial_current_scale_p,
    potential_scale,
    tau_diffusion_e,
)

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Electrolyte Properties
M_w = pybamm.Parameter("Molar mass of water")
M_p = pybamm.Parameter("Molar mass of cations")
M_n = pybamm.Parameter("Molar mass of anions")
M_e = M_n + M_p  # Molar mass of electrolyte [kg.mol-1]
DeltaVliq_n = (
    V_minus - V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
DeltaVliq_p = (
    2 * V_w - V_minus - 3 * V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

# Electrode Properties
V_Pb = pybamm.Parameter("Molar volume of lead")
V_PbO2 = pybamm.Parameter("Molar volume of lead dioxide")
V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate")
DeltaVsurf_n = V_Pb - V_PbSO4  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurf_p = V_PbSO4 - V_PbO2  # Net Molar Volume consumed in pos electrode [m3.mol-1]
d = pybamm.Parameter("Pore size")
eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
eps_s_max = pybamm.Parameter("Maximum porosity of separator")
eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")

# --------------------------------------------------------------------------------------
"""Functions"""

rho_dim = pybamm.Parameter("epsn_max")
mu_dim = pybamm.Parameter("epsn_max")


def U_n(c_en):
    "Dimensionless open-circuit potential in the negative electrode"
    c_en_dimensional = c_en * c_e_typ
    return (U_n_dimensional(c_en_dimensional) - U_n_ref) / potential_scale


def U_p(c_ep):
    "Dimensionless open-circuit potential in the positive electrode"
    c_ep_dimensional = c_ep * c_e_typ
    return (U_p_dimensional(c_ep_dimensional) - U_p_ref) / potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""

tau_discharge = F * c_e_typ * L_x / i_typ
velocity_scale = i_typ / (c_e_typ * F)  # Reaction velocity scale

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Electrolyte Properties
beta_surf_n = -c_e_typ * DeltaVsurf_n / ne_n  # Molar volume change (lead)
beta_surf_p = -c_e_typ * DeltaVsurf_p / ne_p  # Molar volume change (lead dioxide)
beta_liq_n = -c_e_typ * DeltaVliq_n / ne_n  # Molar volume change (electrolyte, neg)
beta_liq_p = -c_e_typ * DeltaVliq_p / ne_p  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    c_e_typ * M_e / rho_dim * (1 - M_w * V_e / V_w * M_e)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    c_e_typ * M_e / rho_dim * (t_plus + M_n / M_e)
)  # Migrative kinematic relationship coefficient
C_e = tau_diffusion_e / tau_discharge
pi_os = (
    mu_dim * velocity_scale * L_x / (d ** 2 * R * T * c_e_typ)
)  # Ratio of viscous pressure scale to osmotic pressure scale

# Electrochemical Reactions
gamma_dl_n = C_dl * potential_scale / interfacial_current_scale_n / tau_discharge
gamma_dl_p = C_dl * potential_scale / interfacial_current_scale_p / tau_discharge

# Initial conditions
q_init = pybamm.Parameter("Initial State of Charge")
q_max = (
    (L_n * eps_n_max + L_s * eps_s_max + L_p * eps_p_max) / L_x / (s_p - s_n)
)  # Dimensionless max capacity
epsDelta_n = beta_surf_n / L_n * q_max
epsDelta_p = beta_surf_p / L_p * q_max
c_e_init = q_init
eps_n_init = eps_n_max - epsDelta_n * (1 - q_init)  # Initial pororsity (neg) [-]
eps_s_init = eps_s_max  # Initial pororsity (sep) [-]
eps_p_init = eps_p_max - epsDelta_p * (1 - q_init)  # Initial pororsity (pos) [-]


# Concatenated symbols
beta_surf = pybamm.PiecewiseConstant(beta_surf_n, 0, beta_surf_p)
eps_init = pybamm.PiecewiseConstant(eps_n_init, eps_s_init, eps_p_init)
