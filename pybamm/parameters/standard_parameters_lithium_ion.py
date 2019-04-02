#
# Standard parameters for lithium-ion battery models
#
"""
Standard pybamm.Parameters for battery models

Physical Constants
------------------
sp.R
    Ideal gas constant
sp.F
    Faraday's constant
sp.T_ref
    Reference temperature

Microscale Geometry
-------------------
R_n, R_p
    Negative and positive particle radii
sp.a_n_dim, sp.a_p_dim
    Negative and positive electrode surface area densities

Electrolyte Properties
----------------------
ce_typ
    Typical lithium ion concentration in electrolyte
De_typ
    Typical lithium ion diffusivity in the electrolyte

Electrode Properties
--------------------
sigma_n, sigma_p
    Electrical conductivities of the negative and positive electrode
cn_max, cp_max
    Maximum lithium concentration in the negative and positive electrode
D_n_typ, Dp_typ
    Typical diffusivitites in the solid electrode material

Electrochemical Reactions
--------------------------
m_n, m_p
    Reaction rates in negative and positive electrode regions

Electrical
----------
voltage_low_cut, voltage_high_cut
    Low and high voltage cut-offs
I_typ
    Typical current density
Phi_typ
    Typical voltage drop across the cell

Initial Conditions
-------------------
ce0_dimensional
    Initial lithium ion concentration in the electrolyte
cn0_dimensional, cp0_dimensional
    Initial lithium concentration in the negative and positive electrodes
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

sp = pybamm.standard_parameters

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Microscale geometry
R_n = pybamm.Parameter("Negative particle radius")
R_p = pybamm.Parameter("Positive particle radius")


# Electrolyte properties
# (1-2*t_plus) is for Nernst-Planck
# 2*(1-t_plus) for Stefan-Maxwell
# Bizeray et al (2016) "Resolving a discrepancy ..."
# note: this is a function for consistancy with lead-acid
def chi(c_e):
    return 2 * (1 - sp.t_plus)


# Electrode properties
c_n_max = pybamm.Parameter("Maximum concentration in negative electrode")
c_p_max = pybamm.Parameter("Maximum concentration in positive electrode")

# Initial conditions
c_e_init_dimensional = pybamm.Parameter("Initial concentration in electrolyte")
c_n_init_dimensional = pybamm.Parameter("Initial concentration in negative electrode")
c_p_init_dimensional = pybamm.Parameter("Initial concentration in positive electrode")

# --------------------------------------------------------------------------------------
"""Functions"""


def D_n_dimensional(c_n):
    "Dimensional diffusivity in negative particle"
    return pybamm.FunctionParameter("Negative electrode diffusivity", c_n)


def D_n(c_n):
    "Dimensionless negative particle diffusivity"
    c_n_dimensional = c_n * c_n_max
    return D_n_dimensional(c_n_dimensional) / D_n_dimensional(c_n_max)


def D_p_dimensional(c_p):
    "Dimensional diffusivity in positive particle"
    return pybamm.FunctionParameter("Positive electrode diffusivity", c_p)


def D_p(c_p):
    "Dimensionless positive particle diffusivity"
    c_p_dimensional = c_p * c_p_max
    return D_p_dimensional(c_p_dimensional) / D_p_dimensional(c_p_max)


def U_n_dimensional(c_s_n):
    "Dimensional open-circuit voltage in the negative electrode [V]"
    return pybamm.FunctionParameter("Negative electrode OCV", c_s_n)


# Because stochiometries vary over different ranges, it is not obvious what this
# scaling should be in general, without evaluating the OCV. Left at c=0.5 for now
U_n_ref = U_n_dimensional(pybamm.Scalar(0.7))


def U_p_dimensional(c_s_p):
    "Dimensional open-circuit voltage of of the positive electrode [V]"
    return pybamm.FunctionParameter("Positive electrode OCV", c_s_p)


# Because stochiometries vary over different ranges, it is not obvious what this
# scaling should be in general, without evaluating the OCV. Left at c=0.5 for now
U_p_ref = U_p_dimensional(pybamm.Scalar(0.7))


def U_n(c_n):
    "Dimensionless open-circuit sp.potential in the negative electrode"
    sto = c_n
    return (U_n_dimensional(sto) - U_n_ref) / sp.potential_scale


def U_p(c_p):
    "Dimensionless open-circuit sp.potential in the positive electrode"
    sto = c_p
    return (U_p_dimensional(sto) - U_p_ref) / sp.potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""
# for purposes of testing dimensionless parameter sizes, we put i_typ = 24 A/m^2
sp.i_typ = 24

# Timescales
# Discharge timescale
tau_discharge = sp.F * c_n_max * sp.L_x / sp.i_typ

# Particle diffusion timescales
tau_diffusion_n = R_n ** 2 / D_n_dimensional(c_n_max)
tau_diffusion_p = R_p ** 2 / D_p_dimensional(c_p_max)

# Electrolyte Diffusion timescale
tau_diffusion_e = sp.L_x ** 2 / sp.D_e_dimensional(sp.c_e_typ)

# reaction timescales
tau_r_n = sp.F / (sp.m_n_dimensional * sp.a_n_dim * sp.c_e_typ ** 0.5)
tau_r_p = sp.F / (sp.m_p_dimensional * sp.a_p_dim * sp.c_e_typ ** 0.5)

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""
# Timescale ratios
C_n = tau_diffusion_n / tau_discharge  # diffusional C-rate in negative electrode
C_p = tau_diffusion_p / tau_discharge  # diffusional C-rate in positive electrode
C_e = tau_diffusion_e / tau_discharge
C_r_n = tau_r_n / tau_discharge
C_r_p = tau_r_p / tau_discharge

# Microscale geometry
epsilon_n = pybamm.Parameter("Negative electrode porosity")
epsilon_s = pybamm.Parameter("Separator porosity")
epsilon_p = pybamm.Parameter("Positive electrode porosity")
epsilon = pybamm.Concatenation(
    pybamm.Broadcast(epsilon_n, ["negative electrode"]),
    pybamm.Broadcast(epsilon_s, ["separator"]),
    pybamm.Broadcast(epsilon_p, ["positive electrode"]),
)

a_n = sp.a_n_dim * R_n
a_p = sp.a_p_dim * R_p

# Electrode Properties
sigma_n = sp.sigma_n_dimensional * sp.potential_scale / sp.i_typ / sp.L_x
sigma_p = sp.sigma_p_dimensional * sp.potential_scale / sp.i_typ / sp.L_x

# Microscale properties
# Note: gamma_n == 1, so not needed
gamma_p = c_p_max / c_n_max

# Electrolyte Properties
C_e = sp.tau_diffusion_e / tau_discharge  # diffusional C-rate in electrolyte
gamma_e = sp.c_e_typ / c_n_max
beta_surf = 0
s = 1

# Electrochemical Reactions
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
c_e_init = c_e_init_dimensional / sp.c_e_typ
c_n_init = c_n_init_dimensional / c_n_max
c_p_init = c_p_init_dimensional / c_p_max
