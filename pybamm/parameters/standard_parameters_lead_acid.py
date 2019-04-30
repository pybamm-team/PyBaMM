#
# Standard parameters for lead-acid battery models
#
"""
Standard parameters for lead-acid battery models
"""
import pybamm
from scipy import constants

# --------------------------------------------------------------------------------------
"File Layout:"
# 1. Dimensional Parameters
# 2. Dimensional Functions
# 3. Scalings
# 4. Dimensionless Parameters
# 5. Dimensionless Functions


# --------------------------------------------------------------------------------------
"1. Dimensional Parameters"
# Physical constants
R = pybamm.Scalar(constants.R)
F = pybamm.Scalar(constants.physical_constants["Faraday constant"][0])
T_ref = pybamm.Parameter("Reference temperature")

# Macroscale geometry
L_n = pybamm.geometric_parameters.L_n
L_s = pybamm.geometric_parameters.L_s
L_p = pybamm.geometric_parameters.L_p
L_x = pybamm.geometric_parameters.L_x
L_y = pybamm.geometric_parameters.L_y
L_z = pybamm.geometric_parameters.L_z
A_cc = pybamm.geometric_parameters.A_cc

# Electrical
I_typ = pybamm.electrical_parameters.I_typ
Q = pybamm.electrical_parameters.Q
C_rate = pybamm.electrical_parameters.C_rate
n_electrodes_parallel = pybamm.electrical_parameters.n_electrodes_parallel
i_typ = pybamm.electrical_parameters.i_typ
voltage_low_cut_dimensional = pybamm.electrical_parameters.voltage_low_cut_dimensional
voltage_high_cut_dimensional = pybamm.electrical_parameters.voltage_high_cut_dimensional
current_with_time = pybamm.electrical_parameters.current_with_time
dimensional_current_with_time = (
    pybamm.electrical_parameters.dimensional_current_with_time
)

# Electrolyte properties
c_e_typ = pybamm.Parameter("Typical electrolyte concentration")
t_plus = pybamm.Parameter("Cation transference number")
V_w = pybamm.Parameter("Partial molar volume of water")
V_plus = pybamm.Parameter("Partial molar volume of cations")
V_minus = pybamm.Parameter("Partial molar volume of anions")
V_e = V_minus + V_plus  # Partial molar volume of electrolyte [m3.mol-1]
nu_plus = pybamm.Parameter("Cation stoichiometry")
nu_minus = pybamm.Parameter("Anion stoichiometry")
nu = nu_plus + nu_minus

# Electrode properties
sigma_n_dimensional = pybamm.Parameter("Negative electrode conductivity")
sigma_p_dimensional = pybamm.Parameter("Positive electrode conductivity")

# Microstructure
a_n_dim = pybamm.Parameter("Negative electrode surface area density")
a_p_dim = pybamm.Parameter("Positive electrode surface area density")
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
C_dl_dimensional = pybamm.Parameter("Double-layer capacity")


# Electrolyte properties
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

# Electrode properties
V_Pb = pybamm.Parameter("Molar volume of lead")
V_PbO2 = pybamm.Parameter("Molar volume of lead-dioxide")
V_PbSO4 = pybamm.Parameter("Molar volume of lead sulfate")
DeltaVsurf_n = V_Pb - V_PbSO4  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurf_p = V_PbSO4 - V_PbO2  # Net Molar Volume consumed in pos electrode [m3.mol-1]
d = pybamm.Parameter("Pore size")
eps_n_max = pybamm.Parameter("Maximum porosity of negative electrode")
eps_s_max = pybamm.Parameter("Maximum porosity of separator")
eps_p_max = pybamm.Parameter("Maximum porosity of positive electrode")


# --------------------------------------------------------------------------------------
"2. Dimensional Functions"


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
    kappa_scale = F ** 2 * D_e_dimensional(c_e_typ) * c_e_typ / (R * T_ref)
    return kappa_e_dimensional(c_e_dimensional) / kappa_scale


def chi_dimensional(c_e):
    return pybamm.FunctionParameter("Darken thermodynamic factor", c_e)


def c_w_dimensional(c_e):
    """
    Water concentration [mol.m-3], from thermodynamics. c_k in [mol.m-3].
    """
    return (1 - c_e * V_e) / V_w


def rho_dimensional(c_e):
    """
    Dimensional density of electrolyte [kg.m-3], from thermodynamics. c_e in [mol.m-3].

    """
    return M_w / V_w * (1 + (M_e * V_w / M_w - V_e) * c_e)


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
    return pybamm.FunctionParameter("Electrolyte viscosity", c_e)


def U_n_dimensional(c_e):
    "Dimensional open-circuit voltage in the negative electrode [V]"
    return pybamm.FunctionParameter("Negative electrode OCV", m_dimensional(c_e))


def U_p_dimensional(c_e):
    "Dimensional open-circuit voltage in the positive electrode [V]"
    return pybamm.FunctionParameter("Positive electrode OCV", m_dimensional(c_e))


U_n_ref = pybamm.FunctionParameter("Negative electrode OCV", pybamm.Scalar(1))
U_p_ref = pybamm.FunctionParameter("Positive electrode OCV", pybamm.Scalar(1))


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
tau_r_n = 1 / (m_n_dimensional * a_n_dim * c_e_typ ** 0.5)
tau_r_p = 1 / (m_p_dimensional * a_p_dim * c_e_typ ** 0.5)

# Electrolyte diffusion timescale
tau_diffusion_e = L_x ** 2 / D_e_dimensional(c_e_typ)


# --------------------------------------------------------------------------------------
"4. Dimensionless Parameters"

# Macroscale Geometry
l_n = pybamm.geometric_parameters.l_n
l_s = pybamm.geometric_parameters.l_s
l_p = pybamm.geometric_parameters.l_p
l_y = pybamm.geometric_parameters.l_y
l_z = pybamm.geometric_parameters.l_z

# Electrode Properties
sigma_n = sigma_n_dimensional * potential_scale / current_scale / L_x
sigma_p = sigma_p_dimensional * potential_scale / current_scale / L_x

# Electrolyte properties
beta_surf_n = -c_e_typ * DeltaVsurf_n / ne_n  # Molar volume change (lead)
beta_surf_p = -c_e_typ * DeltaVsurf_p / ne_p  # Molar volume change (lead dioxide)
beta_surf = pybamm.Concatenation(
    pybamm.Broadcast(beta_surf_n, ["negative electrode"]),
    pybamm.Broadcast(0, ["separator"]),
    pybamm.Broadcast(beta_surf_p, ["positive electrode"]),
)
beta_liq_n = -c_e_typ * DeltaVliq_n / ne_n  # Molar volume change (electrolyte, neg)
beta_liq_p = -c_e_typ * DeltaVliq_p / ne_p  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    c_e_typ * M_e / rho_dimensional(c_e_typ) * (1 - M_w * V_e / V_w * M_e)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    c_e_typ * M_e / rho_dimensional(c_e_typ) * (t_plus + M_n / M_e)
)  # Migrative kinematic relationship coefficient
C_e = tau_diffusion_e / tau_discharge
pi_os = (
    mu_dimensional(c_e_typ) * velocity_scale * L_x / (d ** 2 * R * T_ref * c_e_typ)
)  # Ratio of viscous pressure scale to osmotic pressure scale
gamma_e = 1  # ratio of electrolyte concentration to electrode concentration, undef.

# Electrochemical reactions
C_dl_n = (
    C_dl_dimensional * potential_scale / interfacial_current_scale_n / tau_discharge
)
C_dl_p = (
    C_dl_dimensional * potential_scale / interfacial_current_scale_p / tau_discharge
)

# Electrochemical Reactions
s_n = -(s_plus_n + ne_n * t_plus) / ne_n  # Dimensionless rection rate (neg)
s_p = -(s_plus_p + ne_p * t_plus) / ne_p  # Dimensionless rection rate (pos)
s = pybamm.Concatenation(
    pybamm.Broadcast(s_n, ["negative electrode"]),
    pybamm.Broadcast(0, ["separator"]),
    pybamm.Broadcast(s_p, ["positive electrode"]),
)
m_n = m_n_dimensional / interfacial_current_scale_n
m_p = m_p_dimensional / interfacial_current_scale_p

# Electrical
voltage_low_cut = (voltage_low_cut_dimensional - (U_p_ref - U_n_ref)) / potential_scale
voltage_high_cut = (
    voltage_high_cut_dimensional - (U_p_ref - U_n_ref)
) / potential_scale

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
eps_init = pybamm.Concatenation(
    pybamm.Broadcast(eps_n_init, ["negative electrode"]),
    pybamm.Broadcast(eps_s_init, ["separator"]),
    pybamm.Broadcast(eps_p_init, ["positive electrode"]),
)


# hack to make consistent ic with lithium-ion
# find a way to not have to do this
c_n_init = c_e_init
c_p_init = c_e_init


# --------------------------------------------------------------------------------------
"5. Dimensionless Functions"


# (1-2*t_plus) is for Nernst-Planck
# 2*(1-t_plus) for Stefan-Maxwell
def chi(c_e):
    c_e_dimensional = c_e * c_e_typ
    alpha = (nu * V_w - V_e) * c_e_typ
    return chi_dimensional(c_e_dimensional) * 2 * (1 - t_plus) / (1 - alpha * c_e)


def c_w(c_e):
    "Dimensionless water concentration"
    return c_w_dimensional(c_e_typ * c_e) / c_w_dimensional(c_e_typ)


def U_n(c_en):
    "Dimensionless open-circuit voltage in the negative electrode"
    c_en_dimensional = c_en * c_e_typ
    return (U_n_dimensional(c_en_dimensional) - U_n_ref) / potential_scale


def U_p(c_ep):
    "Dimensionless open-circuit voltage in the positive electrode"
    c_ep_dimensional = c_ep * c_e_typ
    return (U_p_dimensional(c_ep_dimensional) - U_p_ref) / potential_scale
