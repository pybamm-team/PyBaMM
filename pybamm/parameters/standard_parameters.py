#
# Standard parameters for battery models
#
"""
Standard Parameters for battery models

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

Initial Conditions
-------------------
ce0_dimensional
    Initial lithium ion concentration in the electrolyte
cn0_dimensional, cp0_dimensional
    Initial lithium concentration in the negative and positive electrodes
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from pybamm import Parameter

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Physical Constants
R = Parameter("R")
F = Parameter("F")
T = Parameter("T_ref")

# Macroscale Geometry
Ln = Parameter("Ln")
Ls = Parameter("Ls")
Lp = Parameter("Lp")
Lx = Ln + Ls + Lp

# Microscale Geometry
R_n = Parameter("R_n")
R_p = Parameter("R_p")
a_n = Parameter("a_n")
a_p = Parameter("a_p")

# Electrolyte Properties
ce_typ = Parameter("ce_typ")  # Typical lithium ion concentration in electrolyte
De_typ = Parameter("De_typ")  # Typical electrolyte diffusivity

# Electrode Properties
sigma_n = Parameter("sigma_n")  # Conductivity in negative electrode
sigma_p = Parameter("sigma_p")  # Conductivity in positive electrode
cn_max = Parameter("cn_max")  # Max concentration in negative electrode
cp_max = Parameter("cp_max")  # Max concentration in positive electrode

# Electrochemical Reactions
m_n = Parameter("m_n")  # Reaction rate in negative electrode
m_p = Parameter("m_p")  # Reaction rate in positive electrode

# Electrical
voltage_low_cut = Parameter("voltage_low_cut")  # Lower voltage cut-off
voltage_high_cut = Parameter("voltage_high_cut")  # Upper voltage cut-off
I_typ = Parameter("I_typ")  # Typical current density
icell = I_typ

# Initial Conditions
ce0_dimensional = Parameter("ce0")  # Initial li ion concentration in electrolyte
cn0_dimensional = Parameter("cn0")  # Initial li concentration in neg electrode
cp0_dimensional = Parameter("cp0")  # Initial li concentration in pos electrode

ne = Parameter("ne")
# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Macroscale Geometry
ln = Ln / Lx
ls = Ls / Lx
lp = Lp / Lx

# Microscale Geometry
epsilon_n = Parameter("epsilon_n")  # Electrolyte volume fraction in neg electrode
epsilon_s = Parameter("epsilon_s")  # Electrolyte volume fraction in separator
epsilon_p = Parameter("epsilon_p")  # Electrolyte volume fraction in pos electrode
b = Parameter("b")  # Bruggeman coefficient
beta_n = a_n * R_n
beta_p = a_p * R_p

# Electrolyte Properties
t_plus = Parameter("t_plus")  # cation transference number
delta = (Lx ** 2 / De_typ) * (I_typ / (F * cn_max * Lx))
nu = cn_max / ce_typ

# Initial conditions
ce0 = ce0_dimensional / ce_typ
