#
# Standard parameters for lithium-ion battery models
#
"""
Standard parameters for lithium-ion battery models
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
T_ref = pybamm.Parameter("Reference temperature [K]")

# Macroscale geometry
L_cn = pybamm.geometric_parameters.L_cn
L_n = pybamm.geometric_parameters.L_n
L_s = pybamm.geometric_parameters.L_s
L_p = pybamm.geometric_parameters.L_p
L_cp = pybamm.geometric_parameters.L_cp
L_x = pybamm.geometric_parameters.L_x
L_y = pybamm.geometric_parameters.L_y
L_z = pybamm.geometric_parameters.L_z
A_cc = pybamm.geometric_parameters.A_cc


# Tab geometry
L_tab_n = pybamm.geometric_parameters.L_tab_n
Centre_y_tab_n = pybamm.geometric_parameters.Centre_y_tab_n
Centre_z_tab_n = pybamm.geometric_parameters.Centre_z_tab_n
L_tab_p = pybamm.geometric_parameters.L_tab_p
Centre_y_tab_p = pybamm.geometric_parameters.Centre_y_tab_p
Centre_z_tab_p = pybamm.geometric_parameters.Centre_z_tab_p
A_tab_n = pybamm.geometric_parameters.A_tab_n
A_tab_p = pybamm.geometric_parameters.A_tab_p

# Electrical
I_typ = pybamm.electrical_parameters.I_typ
Q = pybamm.electrical_parameters.Q
C_rate = pybamm.electrical_parameters.C_rate
n_electrodes_parallel = pybamm.electrical_parameters.n_electrodes_parallel
i_typ = pybamm.electrical_parameters.i_typ
voltage_low_cut_dimensional = pybamm.electrical_parameters.voltage_low_cut_dimensional
voltage_high_cut_dimensional = pybamm.electrical_parameters.voltage_high_cut_dimensional
current_with_time = pybamm.electrical_parameters.current_with_time
dimensional_current_density_with_time = (
    pybamm.electrical_parameters.dimensional_current_density_with_time
)

# Electrolyte properties
c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")

# Electrode properties
c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
sigma_cn_dimensional = pybamm.Parameter(
    "Negative current collector conductivity [S.m-1]"
)
sigma_n_dim = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
sigma_p_dim = pybamm.Parameter("Positive electrode conductivity [S.m-1]")
sigma_cp_dimensional = pybamm.Parameter(
    "Positive current collector conductivity [S.m-1]"
)

# Microscale geometry
a_n_dim = pybamm.geometric_parameters.a_n_dim
a_p_dim = pybamm.geometric_parameters.a_p_dim
R_n = pybamm.geometric_parameters.R_n
R_p = pybamm.geometric_parameters.R_p
b = pybamm.geometric_parameters.b

# Electrochemical reactions
m_n_dimensional = pybamm.Parameter(
    "Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
)
m_p_dimensional = pybamm.Parameter(
    "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
)
ne_n = pybamm.Parameter("Negative electrode electrons in reaction")
ne_p = pybamm.Parameter("Positive electrode electrons in reaction")
C_dl_dimensional = pybamm.Parameter("Double-layer capacity [F.m-2]")


# Initial conditions
c_e_init_dimensional = pybamm.Parameter(
    "Initial concentration in electrolyte [mol.m-3]"
)
c_n_init_dimensional = pybamm.Parameter(
    "Initial concentration in negative electrode [mol.m-3]"
)
c_p_init_dimensional = pybamm.Parameter(
    "Initial concentration in positive electrode [mol.m-3]"
)

# thermal
Delta_T = pybamm.thermal_parameters.Delta_T

# --------------------------------------------------------------------------------------
"2. Dimensional Functions"


def D_e_dimensional(c_e):
    "Dimensional diffusivity in electrolyte"
    return pybamm.FunctionParameter("Electrolyte diffusivity", c_e)


def kappa_e_dimensional(c_e):
    "Dimensional electrolyte conductivity"
    return pybamm.FunctionParameter("Electrolyte conductivity", c_e)


def D_n_dimensional(c_n):
    "Dimensional diffusivity in negative particle"
    return pybamm.FunctionParameter("Negative electrode diffusivity", c_n)


def D_p_dimensional(c_p):
    "Dimensional diffusivity in positive particle"
    return pybamm.FunctionParameter("Positive electrode diffusivity", c_p)


def U_n_dimensional(sto):
    "Dimensional open-circuit voltage in the negative electrode [V]"
    return pybamm.FunctionParameter("Negative electrode OCV", sto)


def U_p_dimensional(sto):
    "Dimensional open-circuit voltage in the positive electrode [V]"
    return pybamm.FunctionParameter("Positive electrode OCV", sto)


def dUdT_n_dimensional(sto):
    "Dimensional entropic change of the negative electrode open circuit voltage [V.K-1]"
    return pybamm.FunctionParameter(
        "Negative electrode OCV entropic change", sto, c_n_max
    )


def dUdT_p_dimensional(sto):
    "Dimensional entropic change of the positive electrode open circuit voltage [V.K-1]"
    return pybamm.FunctionParameter(
        "Positive electrode OCV entropic change", sto, c_p_max
    )


# can maybe improve ref value at some stage
U_n_ref = U_n_dimensional(pybamm.Scalar(0.7))

# can maybe improve ref value at some stage
U_p_ref = U_p_dimensional(pybamm.Scalar(0.7))


# -------------------------------------------------------------------------------------
"3. Scales"
# concentration
electrolyte_concentration_scale = c_e_typ
negative_particle_concentration_scale = c_n_max
positive_particle_concentration_scale = c_n_max

# electrical
potential_scale = R * T_ref / F
current_scale = i_typ
interfacial_current_scale_n = i_typ / (a_n_dim * L_x)
interfacial_current_scale_p = i_typ / (a_p_dim * L_x)

# Discharge timescale
tau_discharge = F * c_n_max * L_x / i_typ

# Reaction timescales
tau_r_n = F / (m_n_dimensional * a_n_dim * c_e_typ ** 0.5)
tau_r_p = F / (m_p_dimensional * a_p_dim * c_e_typ ** 0.5)

# Electrolyte diffusion timescale
tau_diffusion_e = L_x ** 2 / D_e_dimensional(c_e_typ)

# Particle diffusion timescales
tau_diffusion_n = R_n ** 2 / D_n_dimensional(c_n_max)
tau_diffusion_p = R_p ** 2 / D_p_dimensional(c_p_max)

# Thermal diffusion timescale
tau_th_yz = pybamm.thermal_parameters.tau_th_yz

# --------------------------------------------------------------------------------------
"4. Dimensionless Parameters"
# Timescale ratios
C_n = tau_diffusion_n / tau_discharge
C_p = tau_diffusion_p / tau_discharge
C_e = tau_diffusion_e / tau_discharge
C_r_n = tau_r_n / tau_discharge
C_r_p = tau_r_p / tau_discharge
C_th = tau_th_yz / tau_discharge

# Concentration ratios
gamma_e = c_e_typ / c_n_max
gamma_p = c_p_max / c_n_max

# Macroscale Geometry
l_cn = pybamm.geometric_parameters.l_cn
l_n = pybamm.geometric_parameters.l_n
l_s = pybamm.geometric_parameters.l_s
l_p = pybamm.geometric_parameters.l_p
l_y = pybamm.geometric_parameters.l_y
l_z = pybamm.geometric_parameters.l_z
l_cp = pybamm.geometric_parameters.l_cp
delta = pybamm.geometric_parameters.delta

# Tab geometry
l_tab_n = pybamm.geometric_parameters.l_tab_n
centre_y_tab_n = pybamm.geometric_parameters.centre_y_tab_n
centre_z_tab_n = pybamm.geometric_parameters.centre_z_tab_n
l_tab_p = pybamm.geometric_parameters.l_tab_p
centre_y_tab_p = pybamm.geometric_parameters.centre_y_tab_p
centre_z_tab_p = pybamm.geometric_parameters.centre_z_tab_p

# Microscale geometry
epsilon_n = pybamm.Parameter("Negative electrode porosity")
epsilon_s = pybamm.Parameter("Separator porosity")
epsilon_p = pybamm.Parameter("Positive electrode porosity")
epsilon = pybamm.Concatenation(
    pybamm.Broadcast(epsilon_n, ["negative electrode"]),
    pybamm.Broadcast(epsilon_s, ["separator"]),
    pybamm.Broadcast(epsilon_p, ["positive electrode"]),
)
a_n = a_n_dim * R_n
a_p = a_p_dim * R_p

# Electrode Properties
sigma_cn = sigma_cn_dimensional * potential_scale / i_typ / L_x
sigma_n = sigma_n_dim * potential_scale / i_typ / L_x
sigma_p = sigma_p_dim * potential_scale / i_typ / L_x
sigma_cp = sigma_cp_dimensional * potential_scale / i_typ / L_x

# should rename this to avoid confusion with Butler-Volmer

alpha = 1 / (sigma_cn * (L_x / L_z) ** 2 * l_cn) + 1 / (
    sigma_cp * (L_x / L_z) ** 2 * l_cp
)

# Electrolyte Properties
t_plus = pybamm.Parameter("Cation transference number")
beta_surf = 0
s = 1 - t_plus


# (1-2*t_plus) is for Nernst-Planck
# 2*(1-t_plus) for Stefan-Maxwell
# Bizeray et al (2016) "Resolving a discrepancy ..."
# note: this is a function for consistancy with lead-acid
def chi(c_e):
    return 2 * (1 - t_plus)


# Electrochemical Reactions
C_dl_n = (
    C_dl_dimensional * potential_scale / interfacial_current_scale_n / tau_discharge
)
C_dl_p = (
    C_dl_dimensional * potential_scale / interfacial_current_scale_p / tau_discharge
)

# Electrical
voltage_low_cut = (voltage_low_cut_dimensional - (U_p_ref - U_n_ref)) / potential_scale
voltage_high_cut = (
    voltage_high_cut_dimensional - (U_p_ref - U_n_ref)
) / potential_scale

# Thermal
rho_cn = pybamm.thermal_parameters.rho_cn
rho_n = pybamm.thermal_parameters.rho_n
rho_s = pybamm.thermal_parameters.rho_s
rho_p = pybamm.thermal_parameters.rho_p
rho_cp = pybamm.thermal_parameters.rho_cp

lambda_cn = pybamm.thermal_parameters.lambda_cn
lambda_n = pybamm.thermal_parameters.lambda_n
lambda_s = pybamm.thermal_parameters.lambda_s
lambda_p = pybamm.thermal_parameters.lambda_p
lambda_cp = pybamm.thermal_parameters.lambda_cp

Theta = pybamm.thermal_parameters.Theta
h = pybamm.thermal_parameters.h

# Initial conditions
c_e_init = c_e_init_dimensional / c_e_typ
c_n_init = c_n_init_dimensional / c_n_max
c_p_init = c_p_init_dimensional / c_p_max
T_init = pybamm.thermal_parameters.T_init


# --------------------------------------------------------------------------------------
"5. Dimensionless Functions"


def D_e(c_e):
    "Dimensionless electrolyte diffusivity"
    c_e_dimensional = c_e * c_e_typ
    return D_e_dimensional(c_e_dimensional) / D_e_dimensional(c_e_typ)


def kappa_e(c_e):
    "Dimensionless electrolyte conductivity"
    c_e_dimensional = c_e * c_e_typ
    kappa_scale = F ** 2 * D_e_dimensional(c_e_typ) * c_e_typ / (R * T_ref)
    return kappa_e_dimensional(c_e_dimensional) / kappa_scale


def D_n(c_s_n):
    "Dimensionless negative particle diffusivity"
    c_s_n_dimensional = c_s_n * c_n_max
    return D_n_dimensional(c_s_n_dimensional) / D_n_dimensional(c_n_max)


def D_p(c_s_p):
    "Dimensionless positive particle diffusivity"
    c_s_p_dimensional = c_s_p * c_p_max
    return D_p_dimensional(c_s_p_dimensional) / D_p_dimensional(c_p_max)


def U_n(c_s_n):
    "Dimensionless open-circuit potential in the negative electrode"
    sto = c_s_n
    return (U_n_dimensional(sto) - U_n_ref) / potential_scale


def U_p(c_s_p):
    "Dimensionless open-circuit potential in the positive electrode"
    sto = c_s_p
    return (U_p_dimensional(sto) - U_p_ref) / potential_scale


def dUdT_n(c_s_n):
    "Dimensionless entropic change in negative open-circuit potential"
    sto = c_s_n
    return dUdT_n_dimensional(sto) * Delta_T / potential_scale


def dUdT_p(c_s_p):
    "Dimensionless entropic change in positive open-circuit potential"
    sto = c_s_p
    return dUdT_p_dimensional(sto) * Delta_T / potential_scale
