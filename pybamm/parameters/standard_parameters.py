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

L_n, Ls, Lp
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

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Physical constants
R = pybamm.Parameter("Ideal gas constant")
F = pybamm.Parameter("Faraday's constant")
T = pybamm.Parameter("Reference temperature")

# Macroscale geometry
L_n = pybamm.Parameter("Negative electrode width")
L_s = pybamm.Parameter("Separator width")
L_p = pybamm.Parameter("Positive electrode width")
L_x = L_n + L_s + L_p  # Total cell width
L_y = pybamm.Parameter("Electrode depth")
L_z = pybamm.Parameter("Electrode height")
A_cc = L_y * L_z  # Area of current collector

# Electrical
I_typ = pybamm.Parameter("Typical current density")
n_electrodes_parallel = pybamm.Parameter(
    "Number of electrodes connected in parallel to make a cell"
)
i_typ = I_typ / (n_electrodes_parallel * A_cc)
Q = pybamm.Parameter("Cell capacity")
Crate = I_typ / Q
voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off")
voltage_high_cut_dimensional = pybamm.Parameter("Upper voltage cut-off")
current_with_time = pybamm.FunctionParameter("current function", pybamm.t)
dimensional_current_with_time = I_typ * current_with_time

# Electrolyte properties
c_e_typ = pybamm.Parameter("Typical electrolyte concentration")
V_w = pybamm.Parameter("Partial molar volume of water")
V_plus = pybamm.Parameter("Partial molar volume of cations")
V_minus = pybamm.Parameter("Partial molar volume of anions")
V_e = V_minus + V_plus  # Partial molar volume of electrolyte [m3.mol-1]
nu_plus = pybamm.Parameter("Cation stoichiometry")
nu_minus = pybamm.Parameter("Anion stoichiometry")
nu = nu_plus + nu_minus
t_plus = pybamm.Parameter("Cation transference number")

# Electrode properties
sigma_n_dimensional = pybamm.Parameter("Negative electrode conductivity")
sigma_p_dimensional = pybamm.Parameter("Positive electrode conductivity")
a_n = pybamm.Parameter("Negative particle surface area density")
a_p = pybamm.Parameter("Positive particle surface area density")
b = pybamm.Parameter("Bruggeman coefficient")

# Electrochemical reactions
m_n_dimensional = pybamm.Parameter(
    "Negative electrode reference exchange-current density"
)
m_p_dimensional = pybamm.Parameter(
    "Positive electrode reference exchange-current density"
)
s_plus_n = pybamm.Parameter("Negative electrode cation signed stoichiometry")
s_plus_p = pybamm.Parameter("Positive electrode cation signed stoichiometry")
ne_n = pybamm.Parameter("Negative electrode electrons in reaction")
ne_p = pybamm.Parameter("Positive electrode electrons in reaction")
C_dl = pybamm.Parameter("Double-layer capacity")
U_n_ref = pybamm.Parameter("Reference OCP in the negative electrode")
U_p_ref = pybamm.Parameter("Reference OCP in the positive electrode")

# -----------------------------------------------------------------------------
"""Functions"""


def D_e_dimensional(c_e):
    "Dimensional diffusivity in electrolyte"
    return pybamm.FunctionParameter("Electrolyte diffusivity", c_e)


def D_e(c_e):
    "Dimensionless electrolyte diffusivity"
    c_e_dimensional = c_e * c_e_typ
    return D_e_dimensional(c_e_dimensional) / D_e_dimensional(c_e_typ)


def kappa_e_dimensional(c_e):
    "Dimensional electrolyte conductivity"
    return pybamm.FunctionParameter("Electrolyte conductivity", c_e)


def kappa_e(c_e):
    "Dimensionless electrolyte conductivity"
    c_e_dimensional = c_e * c_e_typ
    return kappa_e_dimensional(c_e_dimensional) / kappa_e_dimensional(c_e_typ)


def chi_dimensional(c_e):
    return pybamm.FunctionParameter("Darken thermodynamic factor", c_e)


def chi(c_e):
    c_e_dimensional = c_e * c_e_typ
    alpha = (nu * V_w - V_e) * c_e_typ
    return chi_dimensional(c_e_dimensional) * 2 * (1 - t_plus) / (1 - alpha * c_e)


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


# --------------------------------------------------------------------------------------
"""Scales"""
concentration_scale = c_e_typ
length_scale = L_x
potential_scale = R * T / F
current_scale = i_typ
interfacial_current_scale_n = i_typ / (a_n * L_x)
interfacial_current_scale_p = i_typ / (a_p * L_x)

# Timescales
# Reaction timescales
tau_rxn_n = 1 / (m_n_dimensional * a_n * c_e_typ ** 0.5)
tau_rxn_p = 1 / (m_p_dimensional * a_p * c_e_typ ** 0.5)
# Diffusion timescale
tau_diffusion_e = L_x ** 2 / D_e_dimensional(c_e_typ)

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Macroscale Geometry
l_n = L_n / length_scale
l_s = L_s / length_scale
l_p = L_p / length_scale
l_y = L_y / L_z
l_z = L_z / L_z

# Electrochemical Reactions
s_n = -(s_plus_n + ne_n * t_plus) / ne_n  # Dimensionless rection rate (neg)
s_p = -(s_plus_p + ne_p * t_plus) / ne_p  # Dimensionless rection rate (pos)
m_n = m_n_dimensional / interfacial_current_scale_n
m_n = m_p_dimensional / interfacial_current_scale_p
# m_n = time_scale / tau_rxn_n
# m_p = time_scale / tau_rxn_n
s = pybamm.PiecewiseConstant(s_n, 0, s_p)

# Electrode Properties
sigma_n = sigma_n_dimensional * potential_scale / current_scale
sigma_p = sigma_p_dimensional * potential_scale / current_scale

# Electrical
voltage_low_cut = (voltage_low_cut_dimensional - (U_p_ref - U_n_ref)) / potential_scale
voltage_high_cut = (
    voltage_high_cut_dimensional - (U_p_ref - U_n_ref)
) / potential_scale
