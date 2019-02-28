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

rho_dimensional = pybamm.Parameter("epsn_max")
mu_dimensional = pybamm.Parameter("epsn_max")


def U_n(c_en):
    "Dimensionless open-circuit sp.potential in the negative electrode"
    c_en_dimensional = c_en * sp.c_e_typ
    return (sp.U_n_dimensional(c_en_dimensional) - sp.U_n_ref) / sp.potential_scale


def U_p(c_ep):
    "Dimensionless open-circuit sp.potential in the positive electrode"
    c_ep_dimensional = c_ep * sp.c_e_typ
    return (sp.U_p_dimensional(c_ep_dimensional) - sp.U_p_ref) / sp.potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""

tau_discharge = sp.F * sp.c_e_typ * sp.L_x / sp.i_typ
velocity_scale = sp.i_typ / (sp.c_e_typ * sp.F)  # Reaction velocity scale

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Electrolyte properties
beta_surf_n = -sp.c_e_typ * DeltaVsurf_n / sp.ne_n  # Molar volume change (lead)
beta_surf_p = -sp.c_e_typ * DeltaVsurf_p / sp.ne_p  # Molar volume change (lead dioxide)
beta_liq_n = (
    -sp.c_e_typ * DeltaVliq_n / sp.ne_n
)  # Molar volume change (electrolyte, neg)
beta_liq_p = (
    -sp.c_e_typ * DeltaVliq_p / sp.ne_p
)  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    sp.c_e_typ * M_e / rho_dimensional * (1 - M_w * sp.V_e / sp.V_w * M_e)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    sp.c_e_typ * M_e / rho_dimensional * (sp.t_plus + M_n / M_e)
)  # Migrative kinematic relationship coefficient
C_e = sp.tau_diffusion_e / tau_discharge
pi_os = (
    mu_dimensional * velocity_scale * sp.L_x / (d ** 2 * sp.R * sp.T * sp.c_e_typ)
)  # Ratio of viscous pressure scale to osmotic pressure scale

# Electrochemical reactions
gamma_dl_n = (
    sp.C_dl * sp.potential_scale / sp.interfacial_current_scale_n / tau_discharge
)
gamma_dl_p = (
    sp.C_dl * sp.potential_scale / sp.interfacial_current_scale_p / tau_discharge
)

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
