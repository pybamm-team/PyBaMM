#
# Standard parameters for battery models
#
"""
Standard pybamm.Parameters for battery models

Physical Constants
------------------
R
    Ideal gas constant
F
    Faraday's constant
T
    Reference temperature

Macroscale Geometry
-------------------

Ln, Ls, Lp
    The widths of the negative electrode, separator and positive electrode, respectively
Lx
    The width of a single cell
ln, ls, lp
    The dimesionless widths of the negative electrode, separator and positive
    electrode respectively

Microscale Geometry
-------------------
R_n, R_p
    Negative and positive particle radii
a_n, a_p
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
Dn_typ, Dp_typ
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

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Physical Constants
R = pybamm.Parameter("R")
F = pybamm.Parameter("F")
T = pybamm.Parameter("T_ref")

# Macroscale Geometry
Ln = pybamm.Parameter("Ln")
Ls = pybamm.Parameter("Ls")
Lp = pybamm.Parameter("Lp")
Lx = Ln + Ls + Lp

# 3D Geometry
Ly = pybamm.Parameter("Ly")
Lz = pybamm.Parameter("Lz")

# Microscale Geometry
R_n = pybamm.Parameter("R_n")
R_p = pybamm.Parameter("R_p")
a_n = pybamm.Parameter("a_n")
a_p = pybamm.Parameter("a_p")

# Electrolyte Properties
ce_typ = pybamm.Parameter("ce_typ")  # Typical lithium ion concentration in electrolyte
De_typ = pybamm.Parameter("De_typ")  # Typical electrolyte diffusivity

# Electrode Properties
sigma_n = pybamm.Parameter("sigma_n")  # Conductivity in negative electrode
sigma_p = pybamm.Parameter("sigma_p")  # Conductivity in positive electrode
cn_max = pybamm.Parameter("cn_max")  # Max concentration in negative electrode
cp_max = pybamm.Parameter("cp_max")  # Max concentration in positive electrode
Dn_typ = pybamm.Parameter("Dn_typ")  # Typical negative particle diffusivity
Dp_typ = pybamm.Parameter("Dp_typ")  # Typical positive particle diffusivity

# Electrochemical Reactions
m_n_dim = pybamm.Parameter("m_n_dim")  # Reaction rate in negative electrode
m_p_dim = pybamm.Parameter("m_p_dim")  # Reaction rate in positive electrode

# Electrical
voltage_low_cut = pybamm.Parameter("voltage_low_cut")  # Lower voltage cut-off
voltage_high_cut = pybamm.Parameter("voltage_high_cut")  # Upper voltage cut-off
I_typ = pybamm.Parameter("I_typ")  # Typical current density
Phi_typ = pybamm.Parameter("Phi_typ")  # Typical voltage drop
current_with_time = pybamm.FunctionParameter("current function", pybamm.t)
dimensional_current_with_time = I_typ * current_with_time

# Initial Conditions
ce0_dimensional = pybamm.Parameter("ce0")  # Initial li ion concentration in electrolyte
cn0_dimensional = pybamm.Parameter("cn0")  # Initial li concentration in neg electrode
cp0_dimensional = pybamm.Parameter("cp0")  # Initial li concentration in pos electrode

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Macroscale Geometry
ln = Ln / Lx
ls = Ls / Lx
lp = Lp / Lx
ly = Ly / Lz
lz = Lz / Lz

# Microscale Geometry
epsilon_n = pybamm.Parameter(
    "epsilon_n"
)  # Electrolyte volume fraction in neg electrode
epsilon_s = pybamm.Parameter("epsilon_s")  # Electrolyte volume fraction in separator
epsilon_p = pybamm.Parameter(
    "epsilon_p"
)  # Electrolyte volume fraction in pos electrode
b = pybamm.Parameter("b")  # Bruggeman coefficient
beta_n = a_n * R_n
beta_p = a_p * R_p

# Discharge timescale
tau_d = (F * cn_max * Lx) / I_typ

# Particle diffusion timescales
tau_n = R_n ** 2 / Dn_typ
tau_p = R_n ** 2 / Dp_typ

# Reaction timescales
tau_r_n = F / (m_n_dim * a_n * ce_typ ** 0.5)
tau_r_p = F / (m_p_dim * a_p * ce_typ ** 0.5)

# Scaled maximum concentration in positive particle
# Note: C_hat_n == 1, so not needed
C_hat_p = cp_max / cn_max

# Ratio of discharge and solid diffusion timescales
gamma_n = tau_d / tau_n
gamma_p = tau_d / tau_p

# Reaction properties
m_n = tau_d / tau_r_n
m_p = tau_d / tau_r_n

# Electrolyte Properties
t_plus = pybamm.Parameter("t_plus")  # cation transference number
delta = (Lx ** 2 / De_typ) * (I_typ / (F * cn_max * Lx))
nu = cn_max / ce_typ

# Ratio of typical to thermal voltage
Lambda = Phi_typ / (R * T / F)

# Initial conditions
ce0 = ce0_dimensional / ce_typ
cn0 = cn0_dimensional / cn_max
cp0 = cp0_dimensional / cp_max

# -----------------------------------------------------------------------------
"""Functions"""


def D_n_dim(c):
    "Dimensional diffusivity in negative particle"
    return Dn_typ


def D_n(c):
    "Dimensionless negative particle diffusivity"
    return D_n_dim(c) / Dn_typ


def D_p_dim(c):
    "Dimensional diffusivity in positive particle"
    return Dp_typ


def D_p(c):
    "Dimensionless positive particle diffusivity"
    return D_p_dim(c) / Dp_typ


def U_n(c):
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
    return out / Phi_typ


def U_p(c):
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
    return out / Phi_typ
