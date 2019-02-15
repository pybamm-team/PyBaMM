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
import pybamm

# --------------------------------------------------------------------------------------
"""Dimensional pybamm.Parameters"""

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

# Electrochemical Reactions
m_n = pybamm.Parameter("m_n")  # Reaction rate in negative electrode
m_p = pybamm.Parameter("m_p")  # Reaction rate in positive electrode

# Electrical
voltage_low_cut = pybamm.Parameter("voltage_low_cut")  # Lower voltage cut-off
voltage_high_cut = pybamm.Parameter("voltage_high_cut")  # Upper voltage cut-off
I_typ = pybamm.Parameter("I_typ")  # Typical current density

# Initial Conditions
ce0_dimensional = pybamm.Parameter("ce0")  # Initial li ion concentration in electrolyte
cn0_dimensional = pybamm.Parameter("cn0")  # Initial li concentration in neg electrode
cp0_dimensional = pybamm.Parameter("cp0")  # Initial li concentration in pos electrode

# --------------------------------------------------------------------------------------
"""Functions"""


def dimensional_current(current_scale, current_function, t):
    """Returns the dimensional current as a function of time

    pybamm.Parameters
    ----------
    current_scale : :class:`numbers.Number` or :class:`pybamm.Symbol`
        The typical scale for the current
    current_function : python function
        The current function
    t : :class:`pybamm.Time`
        The independent variable "time"

    Returns
    -------
    :class:`pybamm.Symbol`
        The current as a function of time
    """
    t = pybamm.t
    return current_scale * pybamm.Function(current_function, t)


def dimensionless_current(current_function, t):
    """Returns the dimensionless current as a function of time

    pybamm.Parameters
    ----------
    current_function : python function
        The current function
    t : :class:`pybamm.Time`
        The independent variable "time"

    Returns
    -------
    :class:`pybamm.Symbol`
        The current as a function of time
    """
    return pybamm.Function(current_function, t)


# --------------------------------------------------------------------------------------
"""Dimensionless pybamm.Parameters"""

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

# Electrolyte Properties
t_plus = pybamm.Parameter("t_plus")  # cation transference number
delta = (Lx ** 2 / De_typ) * (I_typ / (F * cn_max * Lx))
nu = cn_max / ce_typ

# Initial conditions
ce0 = ce0_dimensional / ce_typ
