#
# Standard parameters for battery models
#
"""
Standard pybamm.Parameters for battery models

Physical Constants
------------------
R
    Ideal gas constant
sp.F
    Faraday's constant
T
    Reference temperature

Macroscale Geometry
-------------------

L_n, L_s, L_p
    The widths of the negative electrode, separator and positive electrode, respectively
L_x
    The width of a single cell
l_n, l_s, l_p
    The dimesionless widths of the negative electrode, separator and positive
    electrode respectively

Microscale Geometry
-------------------
R_n, R_p
    Negative and positive particle radii
sp.a_n, sp.a_p
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
    return 1  # pybamm.FunctionParameter("Negative electrode diffusivity", c_n)


def D_n(c_n):
    "Dimensionless negative particle diffusivity"
    c_n_dimensional = c_n * c_n_max
    return D_n_dimensional(c_n_dimensional) / D_n_dimensional(c_n_max)


def D_p_dimensional(c_p):
    "Dimensional diffusivity in positive particle"
    return 1  # pybamm.FunctionParameter("Positive electrode diffusivity", c_p)


def D_p(c_p):
    "Dimensionless positive particle diffusivity"
    c_p_dimensional = c_p * c_p_max
    return D_p_dimensional(c_p_dimensional) / D_p_dimensional(c_p_max)


def chi(c_e):
    "Dimensionless factor in MacInnes equation"
    return 1 - 2 * sp.t_plus


def U_n_dimensional(c):
    "Dimensionless open circuit potential in the negative electrode"
    #  out = (0.194 + 1.5 * np.exp(-120.0 * c)
    #       + 0.0351 * np.tanh((c - 0.286) / 0.083)
    #       - 0.0045 * np.tanh((c - 0.849) / 0.119)
    #       - 0.035 * np.tanh((c - 0.9233) / 0.05)
    #       - 0.0147 * np.tanh((c - 0.5) / 0.034)
    #       - 0.102 * np.tanh((c - 0.194) / 0.142)
    #       - 0.022 * np.tanh((c - 0.9) / 0.0164)
    #       - 0.011 * np.tanh((c - 0.124) / 0.0226)
    #       + 0.0155 * np.tanh((c - 0.105) / 0.029))
    # Set constant until functions implemented correctly
    out = 0.2230
    return out


U_n_ref = U_n_dimensional(1)


def U_p_dimensional(c):
    "Dimensionless open circuit potential in the positive electrode"
    # stretch = 1.062
    # sto = stretch * c
    # out = (2.16216 + 0.07645 * np.tanh(30.834 - 54.4806 * sto)
    #       + 2.1581 * np.tanh(52.294 - 50.294 * sto)
    #       - 0.14169 * np.tanh(11.0923 - 19.8543 * sto)
    #       + 0.2051 * np.tanh(1.4684 - 5.4888 * sto)
    #       + 0.2531 * np.tanh((-sto + 0.56478) / 0.1316)
    #       - 0.02167 * np.tanh((sto - 0.525) / 0.006))
    # Set constant until functions implemented correctly
    out = 4.1212
    return out


U_p_ref = U_p_dimensional(1)


def U_n(c_n):
    "Dimensionless open-circuit sp.potential in the negative electrode"
    c_n_dimensional = c_n * c_n_max
    return (U_n_dimensional(c_n_dimensional) - U_n_ref) / sp.potential_scale


def U_p(c_p):
    "Dimensionless open-circuit sp.potential in the positive electrode"
    c_p_dimensional = c_p * c_p_max
    return (U_p_dimensional(c_p_dimensional) - U_p_ref) / sp.potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""

# Timescales
# Discharge timescale
tau_discharge = sp.F * c_n_max * sp.L_x / sp.i_typ
# Particle diffusion timescales
tau_diffusion_n = R_n ** 2 / D_n_dimensional(c_n_max)
tau_diffusion_p = R_n ** 2 / D_p_dimensional(c_p_max)

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Microscale geometry
epsilon_n = pybamm.Parameter("Negative electrode porosity")
epsilon_s = pybamm.Parameter("Separator porosity")
epsilon_p = pybamm.Parameter("Positive electrode porosity")
epsilon = pybamm.Concatenation(
    pybamm.Broadcast(epsilon_n, ["negative electrode"]),
    pybamm.Broadcast(epsilon_s, ["separator"]),
    pybamm.Broadcast(epsilon_p, ["positive electrode"]),
)
# QUESTION: can we call these something different? they clash with the betas in Pb-acid
beta_n = sp.a_n * R_n
beta_p = sp.a_p * R_p

# Microscale properties
# Note: gamma_hat_n == 1, so not needed
gamma_hat_p = c_p_max / c_n_max
C_n = tau_discharge / tau_diffusion_n  # diffusional C-rate in negative electrode
C_p = tau_discharge / tau_diffusion_p  # diffusional C-rate in positive electrode

# Electrolyte Properties
C_e = sp.tau_diffusion_e / tau_discharge  # diffusional C-rate in electrolyte
gamma_hat_e = sp.c_e_typ / c_n_max
beta_surf = 0

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
