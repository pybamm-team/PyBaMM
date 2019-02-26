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
from pybamm.standard_parameters import (
    a_n,
    a_p,
    C_dl,
    c_e_typ,
    F,
    i_typ,
    L_x,
    U_n_dimensional,
    U_n_ref,
    U_p_dimensional,
    U_p_ref,
    interfacial_current_scale_n,
    interfacial_current_scale_p,
    potential_scale,
    tau_diffusion_e,
)

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Microscale Geometry
R_n = pybamm.Parameter("Negative particle radius")
R_p = pybamm.Parameter("Positive particle radius")

# Electrode Properties
c_n_max = pybamm.Parameter("Maximum concentration in negative electrode")
c_p_max = pybamm.Parameter("Maximum concentration in positive electrode")

# Initial Conditions
c_e_init_dimensional = pybamm.Parameter("Initial concentration in electrolyte")
c_n_init_dimensional = pybamm.Parameter("Initial concentration in neg electrode")
c_p_init_dimensional = pybamm.Parameter("Initial concentration in pos electrode")

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


def U_n(c_n):
    "Dimensionless open-circuit potential in the negative electrode"
    c_n_dimensional = c_n * c_n_max
    return (U_n_dimensional(c_n_dimensional) - U_n_ref) / potential_scale


def U_p(c_p):
    "Dimensionless open-circuit potential in the positive electrode"
    c_p_dimensional = c_p * c_p_max
    return (U_p_dimensional(c_p_dimensional) - U_p_ref) / potential_scale


# --------------------------------------------------------------------------------------
"""Scales"""

# Timescales
# Discharge timescale
tau_discharge = F * c_n_max * L_x / i_typ
# Particle diffusion timescales
tau_diffusion_n = R_n ** 2 / D_n_dimensional(c_n_max)
tau_diffusion_p = R_n ** 2 / D_p_dimensional(c_p_max)

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

# Microscale Geometry
epsilon_n = pybamm.Parameter("Negative electrode porosity")
epsilon_s = pybamm.Parameter("Separator porosity")
epsilon_p = pybamm.Parameter("Positive electrode porosity")
epsilon = pybamm.PiecewiseConstant(epsilon_n, epsilon_s, epsilon_p)
beta_n = a_n * R_n
beta_p = a_p * R_p

# Microscale Properties
# Note: gamma_hat_n == 1, so not needed
gamma_hat_p = c_p_max / c_n_max
C_n = tau_discharge / tau_diffusion_n  # diffusional C-rate in negative electrode
C_p = tau_discharge / tau_diffusion_p  # diffusional C-rate in positive electrode

# Electrolyte Properties
C_e = tau_diffusion_e / tau_discharge  # diffusional C-rate in electrolyte
nu = c_n_max / c_e_typ

# Electrochemical Reactions
gamma_dl_n = C_dl * potential_scale / interfacial_current_scale_n / tau_discharge
gamma_dl_p = C_dl * potential_scale / interfacial_current_scale_p / tau_discharge

# Initial conditions
c_e_init = c_e_init_dimensional / c_e_typ
c_n_init = c_n_init_dimensional / c_n_max
c_p_init = c_p_init_dimensional / c_p_max
