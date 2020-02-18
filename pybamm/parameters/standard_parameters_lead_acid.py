#
# Standard parameters for lead-acid battery models
#
"""
Standard Parameters for lead-acid battery models
"""
import pybamm
import numpy as np
from scipy import constants


# --------------------------------------------------------------------------------------
"File Layout:"
# 1. Dimensional Parameters
# 2. Dimensional Functions
# 3. Scalings
# 4. Dimensionless Parameters
# 5. Dimensionless Functions
# 6. Input current

# --------------------------------------------------------------------------------------
"1. Dimensional Parameters"
# Physical constants
R = pybamm.Scalar(constants.R)
F = pybamm.Scalar(constants.physical_constants["Faraday constant"][0])
T_ref = pybamm.Parameter("Reference temperature [K]")

# Macroscale geometry
L_n = pybamm.geometric_parameters.L_n
L_s = pybamm.geometric_parameters.L_s
L_p = pybamm.geometric_parameters.L_p
L_x = pybamm.geometric_parameters.L_x
L_y = pybamm.geometric_parameters.L_y
L_z = pybamm.geometric_parameters.L_z
A_cc = pybamm.geometric_parameters.A_cc
W = L_y
H = L_z
A_cs = A_cc
delta = L_x / H

# Electrical
I_typ = pybamm.electrical_parameters.I_typ
Q = pybamm.electrical_parameters.Q
C_rate = pybamm.electrical_parameters.C_rate
n_electrodes_parallel = pybamm.electrical_parameters.n_electrodes_parallel
i_typ = pybamm.electrical_parameters.i_typ
voltage_low_cut_dimensional = pybamm.electrical_parameters.voltage_low_cut_dimensional
voltage_high_cut_dimensional = pybamm.electrical_parameters.voltage_high_cut_dimensional

# Electrolyte properties
c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
t_plus = pybamm.Parameter("Cation transference number")
V_w = pybamm.Parameter("Partial molar volume of water [m3.mol-1]")
V_plus = pybamm.Parameter("Partial molar volume of cations [m3.mol-1]")
V_minus = pybamm.Parameter("Partial molar volume of anions [m3.mol-1]")
V_e = V_minus + V_plus  # Partial molar volume of electrolyte [m3.mol-1]
nu_plus = pybamm.Parameter("Cation stoichiometry")
nu_minus = pybamm.Parameter("Anion stoichiometry")
nu = nu_plus + nu_minus

# Other species properties
c_ox_init_dim = pybamm.Parameter("Initial oxygen concentration [mol.m-3]")
c_ox_typ = c_e_typ  # pybamm.Parameter("Typical oxygen concentration [mol.m-3]")

# Electrode properties
sigma_n_dim = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
sigma_p_dim = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
# In lead-acid the current collector and electrodes are the same (same conductivity)
sigma_cn_dimensional = sigma_n_dim
sigma_cp_dimensional = sigma_p_dim

# Microstructure
a_n_dim = pybamm.geometric_parameters.a_n_dim
a_p_dim = pybamm.geometric_parameters.a_p_dim
b_e_n = pybamm.geometric_parameters.b_e_n
b_e_s = pybamm.geometric_parameters.b_e_s
b_e_p = pybamm.geometric_parameters.b_e_p
b_s_n = pybamm.geometric_parameters.b_s_n
b_s_s = pybamm.geometric_parameters.b_s_s
b_s_p = pybamm.geometric_parameters.b_s_p
xi_n = pybamm.Parameter("Negative electrode morphological parameter")
xi_p = pybamm.Parameter("Positive electrode morphological parameter")
# no binder
epsilon_inactive_n = pybamm.Scalar(0)
epsilon_inactive_s = pybamm.Scalar(0)
epsilon_inactive_p = pybamm.Scalar(0)

# Electrochemical reactions
# Main
j0_n_S_ref_dimensional = pybamm.Parameter(
    "Negative electrode reference exchange-current density [A.m-2]"
)
j0_p_S_ref_dimensional = pybamm.Parameter(
    "Positive electrode reference exchange-current density [A.m-2]"
)
s_plus_n_S_dim = pybamm.Parameter("Negative electrode cation signed stoichiometry")
s_plus_p_S_dim = pybamm.Parameter("Positive electrode cation signed stoichiometry")
ne_n_S = pybamm.Parameter("Negative electrode electrons in reaction")
ne_p_S = pybamm.Parameter("Positive electrode electrons in reaction")
C_dl_n_dimensional = pybamm.Parameter(
    "Negative electrode double-layer capacity [F.m-2]"
)
C_dl_p_dimensional = pybamm.Parameter(
    "Positive electrode double-layer capacity [F.m-2]"
)
# Oxygen
j0_n_Ox_ref_dimensional = pybamm.Parameter(
    "Negative electrode reference exchange-current density (oxygen) [A.m-2]"
)
j0_p_Ox_ref_dimensional = pybamm.Parameter(
    "Positive electrode reference exchange-current density (oxygen) [A.m-2]"
)
s_plus_Ox_dim = pybamm.Parameter("Signed stoichiometry of cations (oxygen reaction)")
s_w_Ox_dim = pybamm.Parameter("Signed stoichiometry of water (oxygen reaction)")
s_ox_Ox_dim = pybamm.Parameter("Signed stoichiometry of oxygen (oxygen reaction)")
ne_Ox = pybamm.Parameter("Electrons in oxygen reaction")
U_Ox_dim = pybamm.Parameter("Oxygen reference OCP vs SHE [V]")
# Hydrogen
j0_n_Hy_ref_dimensional = pybamm.Parameter(
    "Negative electrode reference exchange-current density (hydrogen) [A.m-2]"
)
j0_p_Hy_ref_dimensional = pybamm.Parameter(
    "Positive electrode reference exchange-current density (hydrogen) [A.m-2]"
)
s_plus_Hy_dim = pybamm.Parameter("Signed stoichiometry of cations (hydrogen reaction)")
s_hy_Hy_dim = pybamm.Parameter("Signed stoichiometry of hydrogen (hydrogen reaction)")
ne_Hy = pybamm.Parameter("Electrons in hydrogen reaction")
U_Hy_dim = pybamm.Parameter("Hydrogen reference OCP vs SHE [V]")


# Electrolyte properties
M_w = pybamm.Parameter("Molar mass of water [kg.mol-1]")
M_plus = pybamm.Parameter("Molar mass of cations [kg.mol-1]")
M_minus = pybamm.Parameter("Molar mass of anions [kg.mol-1]")
M_e = M_minus + M_plus  # Molar mass of electrolyte [kg.mol-1]

DeltaVliq_n = (
    V_minus - V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
DeltaVliq_p = (
    2 * V_w - V_minus - 3 * V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

# Other species properties
D_ox_dimensional = pybamm.Parameter("Oxygen diffusivity [m2.s-1]")
D_hy_dimensional = pybamm.Parameter("Hydrogen diffusivity [m2.s-1]")
V_ox = pybamm.Parameter("Partial molar volume of oxygen molecules [m3.mol-1]")
V_hy = pybamm.Parameter("Partial molar volume of hydrogen molecules [m3.mol-1]")
M_ox = pybamm.Parameter("Molar mass of oxygen molecules [kg.mol-1]")
M_hy = pybamm.Parameter("Molar mass of hydrogen molecules [kg.mol-1]")

# Electrode properties
V_Pb = pybamm.Parameter("Molar volume of lead [m3.mol-1]")
V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide [m3.mol-1]")
V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate [m3.mol-1]")
DeltaVsurf_n = V_Pb - V_PbSO4  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurf_p = V_PbSO4 - V_PbO2  # Net Molar Volume consumed in pos electrode [m3.mol-1]
d_n = pybamm.Parameter("Negative electrode pore size [m]")
d_p = pybamm.Parameter("Positive electrode pore size [m]")
eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
eps_s_max = pybamm.Parameter("Maximum porosity of separator")
eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")
Q_n_max_dimensional = pybamm.Parameter("Negative electrode volumetric capacity [C.m-3]")
Q_p_max_dimensional = pybamm.Parameter("Positive electrode volumetric capacity [C.m-3]")


# Fake thermal
Delta_T = pybamm.Scalar(0)


# --------------------------------------------------------------------------------------
"2. Dimensional Functions"


def D_e_dimensional(c_e, T):
    "Dimensional diffusivity in electrolyte"
    return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", c_e)


def kappa_e_dimensional(c_e, T):
    "Dimensional electrolyte conductivity"
    return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", c_e)


def chi_dimensional(c_e):
    return pybamm.FunctionParameter("Darken thermodynamic factor", c_e)


def c_w_dimensional(c_e, c_ox=0, c_hy=0):
    """
    Water concentration [mol.m-3], from thermodynamics. c_k in [mol.m-3].
    """
    return (1 - c_e * V_e - c_ox * V_ox - c_hy * V_hy) / V_w


def c_T(c_e, c_ox=0, c_hy=0):
    """
    Total liquid molarity [mol.m-3], from thermodynamics. c_k in [mol.m-3].
    """
    return (1 + (2 * V_w - V_e) * c_e + (V_w - V_ox) * c_ox + (V_w - V_hy) * c_hy) / V_w


def rho_dimensional(c_e, c_ox=0, c_hy=0):
    """
    Dimensional density of electrolyte [kg.m-3], from thermodynamics. c_k in [mol.m-3].
    """
    return (
        M_w / V_w
        + (M_e - V_e * M_w / V_w) * c_e
        + (M_ox - V_ox * M_w / V_w) * c_ox
        + (M_hy - V_hy * M_w / V_w) * c_hy
    )


def m_dimensional(c_e):
    """
    Dimensional electrolyte molar mass [mol.kg-1], from thermodynamics.
    c_e in [mol.m-3].
    """
    return c_e * V_w / ((1 - c_e * V_e) * M_w)


def mu_dimensional(c_e):
    """
    Dimensional viscosity of electrolyte [kg.m-1.s-1].
    """
    return pybamm.FunctionParameter("Electrolyte viscosity [kg.m-1.s-1]", c_e)


def U_n_dimensional(c_e, T):
    "Dimensional open-circuit voltage in the negative electrode [V]"
    return pybamm.FunctionParameter(
        "Negative electrode open-circuit potential [V]", m_dimensional(c_e)
    )


def U_p_dimensional(c_e, T):
    "Dimensional open-circuit voltage in the positive electrode [V]"
    return pybamm.FunctionParameter(
        "Positive electrode open-circuit potential [V]", m_dimensional(c_e)
    )


D_e_typ = D_e_dimensional(c_e_typ, T_ref)
rho_typ = rho_dimensional(c_e_typ)
mu_typ = mu_dimensional(c_e_typ)
U_n_ref = pybamm.FunctionParameter(
    "Negative electrode open-circuit potential [V]", pybamm.Scalar(1)
)
U_p_ref = pybamm.FunctionParameter(
    "Positive electrode open-circuit potential [V]", pybamm.Scalar(1)
)


# --------------------------------------------------------------------------------------
"3. Scales"

# concentrations
electrolyte_concentration_scale = c_e_typ

# electrical
potential_scale = R * T_ref / F
current_scale = i_typ
interfacial_current_scale_n = i_typ / (a_n_dim * L_x)
interfacial_current_scale_p = i_typ / (a_p_dim * L_x)

velocity_scale = i_typ / (c_e_typ * F)  # Reaction velocity scale

# Discharge timescale
tau_discharge = F * c_e_typ * L_x / i_typ

# Reaction timescales
# should this be * F?
tau_r_n = 1 / (j0_n_S_ref_dimensional * a_n_dim * c_e_typ ** 0.5)
tau_r_p = 1 / (j0_p_S_ref_dimensional * a_p_dim * c_e_typ ** 0.5)

# Electrolyte diffusion timescale
tau_diffusion_e = L_x ** 2 / D_e_typ

# Choose discharge timescale
timescale = tau_discharge

# --------------------------------------------------------------------------------------
"4. Dimensionless Parameters"

# Macroscale Geometry
l_n = pybamm.geometric_parameters.l_n
l_s = pybamm.geometric_parameters.l_s
l_p = pybamm.geometric_parameters.l_p
l_y = pybamm.geometric_parameters.l_y
l_z = pybamm.geometric_parameters.l_z
# In lead-acid the current collector and electrodes are the same (same thickness)
l_cn = l_n
l_cp = l_p

# Tab geometry
l_tab_n = pybamm.geometric_parameters.l_tab_n
centre_y_tab_n = pybamm.geometric_parameters.centre_y_tab_n
centre_z_tab_n = pybamm.geometric_parameters.centre_z_tab_n
l_tab_p = pybamm.geometric_parameters.l_tab_p
centre_y_tab_p = pybamm.geometric_parameters.centre_y_tab_p
centre_z_tab_p = pybamm.geometric_parameters.centre_z_tab_p

# Diffusive kinematic relationship coefficient
omega_i = c_e_typ * M_e / rho_typ * (t_plus + M_minus / M_e)
# Migrative kinematic relationship coefficient (electrolyte)
omega_c_e = c_e_typ * M_e / rho_typ * (1 - M_w * V_e / V_w * M_e)
C_e = tau_diffusion_e / tau_discharge
# Ratio of viscous pressure scale to osmotic pressure scale (electrolyte)
pi_os_e = mu_typ * velocity_scale * L_x / (d_n ** 2 * R * T_ref * c_e_typ)
# ratio of electrolyte concentration to electrode concentration, undefined
gamma_e = pybamm.Scalar(1)
# Reynolds number
Re = rho_typ * velocity_scale * L_x / mu_typ

# Other species properties
# Oxygen
curlyD_ox = D_ox_dimensional / D_e_typ
omega_c_ox = c_e_typ * M_ox / rho_typ * (1 - M_w * V_ox / V_w * M_ox)
# Hydrogen
curlyD_hy = D_hy_dimensional / D_e_typ
omega_c_hy = c_e_typ * M_hy / rho_typ * (1 - M_w * V_hy / V_w * M_hy)

# Electrode Properties
sigma_cn = sigma_cn_dimensional * potential_scale / i_typ / L_x
sigma_n = sigma_n_dim * potential_scale / current_scale / L_x
sigma_p = sigma_p_dim * potential_scale / current_scale / L_x
sigma_cp = sigma_cp_dimensional * potential_scale / i_typ / L_x
sigma_n_prime = sigma_n * delta ** 2
sigma_p_prime = sigma_p * delta ** 2
delta_pore_n = 1 / (a_n_dim * L_x)
delta_pore_p = 1 / (a_p_dim * L_x)
Q_n_max = Q_n_max_dimensional / (c_e_typ * F)
Q_p_max = Q_p_max_dimensional / (c_e_typ * F)
beta_U_n = 1 / Q_n_max
beta_U_p = -1 / Q_p_max

# Electrochemical reactions
# Main
s_plus_n_S = s_plus_n_S_dim / ne_n_S
s_plus_p_S = s_plus_p_S_dim / ne_p_S
s_n = -(s_plus_n_S + t_plus)  # Dimensionless rection rate (neg)
s_p = -(s_plus_p_S + t_plus)  # Dimensionless rection rate (pos)
s = pybamm.Concatenation(
    pybamm.FullBroadcast(s_n, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(0, ["separator"], "current collector"),
    pybamm.FullBroadcast(s_p, ["positive electrode"], "current collector"),
)
j0_n_S_ref = j0_n_S_ref_dimensional / interfacial_current_scale_n
j0_p_S_ref = j0_p_S_ref_dimensional / interfacial_current_scale_p
C_dl_n = (
    C_dl_n_dimensional * potential_scale / interfacial_current_scale_n / tau_discharge
)
C_dl_p = (
    C_dl_p_dimensional * potential_scale / interfacial_current_scale_p / tau_discharge
)
ne_n = ne_n_S
ne_p = ne_p_S
# Oxygen
s_plus_Ox = s_plus_Ox_dim / ne_Ox
s_w_Ox = s_w_Ox_dim / ne_Ox
s_ox_Ox = s_ox_Ox_dim / ne_Ox
j0_n_Ox_ref = j0_n_Ox_ref_dimensional / interfacial_current_scale_n
j0_p_Ox_ref = j0_p_Ox_ref_dimensional / interfacial_current_scale_p
U_n_Ox = (U_Ox_dim - U_n_ref) / potential_scale
U_p_Ox = (U_Ox_dim - U_p_ref) / potential_scale
# Hydrogen
s_plus_Hy = s_plus_Hy_dim / ne_Hy
s_hy_Hy = s_hy_Hy_dim / ne_Hy
j0_n_Hy_ref = j0_n_Hy_ref_dimensional / interfacial_current_scale_n
j0_p_Hy_ref = j0_p_Hy_ref_dimensional / interfacial_current_scale_p
U_n_Hy = (U_Hy_dim - U_n_ref) / potential_scale
U_p_Hy = (U_Hy_dim - U_p_ref) / potential_scale

# Electrolyte properties
beta_surf_n = -c_e_typ * DeltaVsurf_n / ne_n_S  # Molar volume change (lead)
beta_surf_p = -c_e_typ * DeltaVsurf_p / ne_p_S  # Molar volume change (lead dioxide)
beta_surf = pybamm.Concatenation(
    pybamm.FullBroadcast(beta_surf_n, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(0, ["separator"], "current collector"),
    pybamm.FullBroadcast(beta_surf_p, ["positive electrode"], "current collector"),
)
beta_liq_n = -c_e_typ * DeltaVliq_n / ne_n_S  # Molar volume change (electrolyte, neg)
beta_liq_p = -c_e_typ * DeltaVliq_p / ne_p_S  # Molar volume change (electrolyte, pos)
beta_n = (beta_surf_n + beta_liq_n) * pybamm.Parameter("Volume change factor")
beta_p = (beta_surf_p + beta_liq_p) * pybamm.Parameter("Volume change factor")
beta = pybamm.Concatenation(
    pybamm.FullBroadcast(beta_n, "negative electrode", "current collector"),
    pybamm.FullBroadcast(0, "separator", "current collector"),
    pybamm.FullBroadcast(beta_p, "positive electrode", "current collector"),
)
beta_Ox = -c_e_typ * (s_plus_Ox * V_plus + s_w_Ox * V_w + s_ox_Ox * V_ox)
beta_Hy = -c_e_typ * (s_plus_Hy * V_plus + s_hy_Hy * V_hy)

# Electrical
voltage_low_cut = (voltage_low_cut_dimensional - (U_p_ref - U_n_ref)) / potential_scale
voltage_high_cut = (
    voltage_high_cut_dimensional - (U_p_ref - U_n_ref)
) / potential_scale

# Electrolyte volumetric capacity
Q_e_max = (l_n * eps_n_max + l_s * eps_s_max + l_p * eps_p_max) / (s_p - s_n)
Q_e_max_dimensional = Q_e_max * c_e_typ * F
capacity = Q_e_max_dimensional * n_electrodes_parallel * A_cs * L_x

# Initial conditions
q_init = pybamm.Parameter("Initial State of Charge")
c_e_init = q_init
c_ox_init = c_ox_init_dim / c_ox_typ
epsilon_n_init = eps_n_max - beta_surf_n * Q_e_max / l_n * (1 - q_init)
epsilon_s_init = eps_s_max
epsilon_p_init = eps_p_max + beta_surf_p * Q_e_max / l_p * (1 - q_init)
epsilon_init = pybamm.Concatenation(
    pybamm.FullBroadcast(epsilon_n_init, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(epsilon_s_init, ["separator"], "current collector"),
    pybamm.FullBroadcast(epsilon_p_init, ["positive electrode"], "current collector"),
)
curlyU_n_init = Q_e_max * (1.2 - q_init) / (Q_n_max * l_n)
curlyU_p_init = Q_e_max * (1.2 - q_init) / (Q_p_max * l_p)


# hack to make consistent ic with lithium-ion
def c_n_init(x):
    return c_e_init


def c_p_init(x):
    return c_e_init


# Thermal effects not implemented for lead-acid, but parameters needed for consistency
T_init = pybamm.Scalar(0)
Theta = pybamm.Scalar(0)  # ratio of typical temperature change to ambient temperature


# --------------------------------------------------------------------------------------
"5. Dimensionless Functions"


def D_e(c_e, T):
    "Dimensionless electrolyte diffusivity"
    c_e_dimensional = c_e * c_e_typ
    return D_e_dimensional(c_e_dimensional, T_ref) / D_e_typ


def kappa_e(c_e, T):
    "Dimensionless electrolyte conductivity"
    c_e_dimensional = c_e * c_e_typ
    kappa_scale = F ** 2 * D_e_typ * c_e_typ / (R * T_ref)
    return kappa_e_dimensional(c_e_dimensional, T_ref) / kappa_scale


# (1-2*t_plus) is for Nernst-Planck
# 2*(1-t_plus) for Stefan-Maxwell
def chi(c_e, c_ox=0, c_hy=0):
    return (
        chi_dimensional(c_e_typ * c_e)
        * (2 * (1 - t_plus))
        / (V_w * c_T(c_e_typ * c_e, c_e_typ * c_ox, c_e_typ * c_hy))
    )


def c_w(c_e):
    "Dimensionless water concentration"
    return c_w_dimensional(c_e_typ * c_e) / c_w_dimensional(c_e_typ)


def m_n(T):
    "Dimensionless negative electrode reaction rate"
    return 1


def m_p(T):
    "Dimensionless positive electrode reaction rate"
    return 1


def U_n(c_e_n, T):
    "Dimensionless open-circuit voltage in the negative electrode"
    c_e_n_dimensional = c_e_n * c_e_typ
    return (U_n_dimensional(c_e_n_dimensional, T_ref) - U_n_ref) / potential_scale


def U_p(c_e_p, T):
    "Dimensionless open-circuit voltage in the positive electrode"
    c_e_p_dimensional = c_e_p * c_e_typ
    return (U_p_dimensional(c_e_p_dimensional, T_ref) - U_p_ref) / potential_scale


# --------------------------------------------------------------------------------------
# 6. Input current and voltage

dimensional_current_with_time = pybamm.FunctionParameter(
    "Current function [A]", pybamm.t * timescale
)
dimensional_current_density_with_time = dimensional_current_with_time / (
    n_electrodes_parallel * pybamm.geometric_parameters.A_cc
)
current_with_time = (
    dimensional_current_with_time / I_typ * pybamm.Function(np.sign, I_typ)
)

