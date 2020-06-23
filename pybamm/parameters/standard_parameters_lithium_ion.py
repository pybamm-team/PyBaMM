#
# Standard parameters for lithium-ion battery models
#
"""
Standard parameters for lithium-ion battery models
"""
import pybamm
import numpy as np


# --------------------------------------------------------------------------------------
"File Layout:"
# 1. Dimensional Parameters
# 2. Dimensional Functions
# 3. Scalings
# 4. Dimensionless Parameters
# 5. Dimensionless Functions
# 6. Input Current

# --------------------------------------------------------------------------------------
"1. Dimensional Parameters"

# Physical constants
R = pybamm.constants.R
F = pybamm.constants.F
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
L = pybamm.geometric_parameters.L
A_cc = pybamm.geometric_parameters.A_cc
A_cooling = pybamm.geometric_parameters.A_cooling
V_cell = pybamm.geometric_parameters.V_cell

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
a_k_dim = pybamm.Concatenation(
    pybamm.FullBroadcast(a_n_dim, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(0, ["separator"], "current collector"),
    pybamm.FullBroadcast(a_p_dim, ["positive electrode"], "current collector"),
)
R_n = pybamm.geometric_parameters.R_n
R_p = pybamm.geometric_parameters.R_p
b_e_n = pybamm.geometric_parameters.b_e_n
b_e_s = pybamm.geometric_parameters.b_e_s
b_e_p = pybamm.geometric_parameters.b_e_p
b_s_n = pybamm.geometric_parameters.b_s_n
b_s_s = pybamm.geometric_parameters.b_s_s
b_s_p = pybamm.geometric_parameters.b_s_p

# Electrochemical reactions
ne_n = pybamm.Parameter("Negative electrode electrons in reaction")
ne_p = pybamm.Parameter("Positive electrode electrons in reaction")
C_dl_n_dimensional = pybamm.Parameter(
    "Negative electrode double-layer capacity [F.m-2]"
)
C_dl_p_dimensional = pybamm.Parameter(
    "Positive electrode double-layer capacity [F.m-2]"
)


# Initial conditions
c_e_init_dimensional = pybamm.Parameter(
    "Initial concentration in electrolyte [mol.m-3]"
)


def c_n_init_dimensional(x):
    "Initial concentration as a function of dimensionless position x"
    inputs = {"Dimensionless through-cell position (x_n)": x}
    return pybamm.FunctionParameter(
        "Initial concentration in negative electrode [mol.m-3]", inputs
    )


def c_p_init_dimensional(x):
    "Initial concentration as a function of dimensionless position x"
    inputs = {"Dimensionless through-cell position (x_p)": x}
    return pybamm.FunctionParameter(
        "Initial concentration in positive electrode [mol.m-3]", inputs
    )


# thermal
Delta_T = pybamm.thermal_parameters.Delta_T

# velocity scale
velocity_scale = pybamm.Scalar(1)

# --------------------------------------------------------------------------------------
"2. Dimensional Functions"


def D_e_dimensional(c_e, T):
    "Dimensional diffusivity in electrolyte"
    inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
    return pybamm.FunctionParameter("Electrolyte diffusivity [m2.s-1]", inputs)


def kappa_e_dimensional(c_e, T):
    "Dimensional electrolyte conductivity"
    inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
    return pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)


def D_n_dimensional(sto, T):
    """Dimensional diffusivity in negative particle. Note this is defined as a
    function of stochiometry"""
    inputs = {"Negative particle stoichiometry": sto, "Temperature [K]": T}
    return pybamm.FunctionParameter("Negative electrode diffusivity [m2.s-1]", inputs)


def D_p_dimensional(sto, T):
    """Dimensional diffusivity in positive particle. Note this is defined as a
    function of stochiometry"""
    inputs = {"Positive particle stoichiometry": sto, "Temperature [K]": T}
    return pybamm.FunctionParameter("Positive electrode diffusivity [m2.s-1]", inputs)


def j0_n_dimensional(c_e, c_s_surf, T):
    "Dimensional negative exchange-current density [A.m-2]"
    inputs = {
        "Electrolyte concentration [mol.m-3]": c_e,
        "Negative particle surface concentration [mol.m-3]": c_s_surf,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Negative electrode exchange-current density [A.m-2]", inputs
    )


def j0_p_dimensional(c_e, c_s_surf, T):
    "Dimensional negative exchange-current density [A.m-2]"
    inputs = {
        "Electrolyte concentration [mol.m-3]": c_e,
        "Positive particle surface concentration [mol.m-3]": c_s_surf,
        "Temperature [K]": T,
    }
    return pybamm.FunctionParameter(
        "Positive electrode exchange-current density [A.m-2]", inputs
    )


def dUdT_n_dimensional(sto):
    """
    Dimensional entropic change of the negative electrode open-circuit potential [V.K-1]
    """
    inputs = {
        "Negative particle stoichiometry": sto,
        "Max negative particle concentration [mol.m-3]": c_n_max,
    }
    return pybamm.FunctionParameter(
        "Negative electrode OCP entropic change [V.K-1]", inputs
    )


def dUdT_p_dimensional(sto):
    """
    Dimensional entropic change of the positive electrode open-circuit potential [V.K-1]
    """
    inputs = {
        "Positive particle stoichiometry": sto,
        "Max positive particle concentration [mol.m-3]": c_p_max,
    }
    return pybamm.FunctionParameter(
        "Positive electrode OCP entropic change [V.K-1]", inputs
    )


def U_n_dimensional(sto, T):
    "Dimensional open-circuit potential in the negative electrode [V]"
    inputs = {"Negative particle stoichiometry": sto}
    u_ref = pybamm.FunctionParameter("Negative electrode OCP [V]", inputs)
    return u_ref + (T - T_ref) * dUdT_n_dimensional(sto)


def U_p_dimensional(sto, T):
    "Dimensional open-circuit potential in the positive electrode [V]"
    inputs = {"Positive particle stoichiometry": sto}
    u_ref = pybamm.FunctionParameter("Positive electrode OCP [V]", inputs)
    return u_ref + (T - T_ref) * dUdT_p_dimensional(sto)


# Reference OCP based on initial concentration at current collector/electrode interface
sto_n_init = c_n_init_dimensional(0) / c_n_max
U_n_ref = U_n_dimensional(sto_n_init, T_ref)

# Reference OCP based on initial concentration at current collector/electrode interface
sto_p_init = c_p_init_dimensional(1) / c_p_max
U_p_ref = U_p_dimensional(sto_p_init, T_ref)

j0_n_ref_dimensional = j0_n_dimensional(c_e_typ, c_n_max / 2, T_ref) * 2
j0_p_ref_dimensional = j0_p_dimensional(c_e_typ, c_p_max / 2, T_ref) * 2

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
tau_r_n = F * c_n_max / (j0_n_ref_dimensional * a_n_dim)
tau_r_p = F * c_p_max / (j0_p_ref_dimensional * a_p_dim)

# Electrolyte diffusion timescale
D_e_typ = D_e_dimensional(c_e_typ, T_ref)
tau_diffusion_e = L_x ** 2 / D_e_typ

# Particle diffusion timescales
tau_diffusion_n = R_n ** 2 / D_n_dimensional(pybamm.Scalar(1), T_ref)
tau_diffusion_p = R_p ** 2 / D_p_dimensional(pybamm.Scalar(1), T_ref)

# Thermal diffusion timescale
tau_th_yz = pybamm.thermal_parameters.tau_th_yz

# Choose discharge timescale
timescale = tau_discharge

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
l_cp = pybamm.geometric_parameters.l_cp
l_x = pybamm.geometric_parameters.l_x
l_y = pybamm.geometric_parameters.l_y
l_z = pybamm.geometric_parameters.l_z
a_cc = pybamm.geometric_parameters.a_cc
a_cooling = pybamm.geometric_parameters.a_cooling
v_cell = pybamm.geometric_parameters.v_cell
l = pybamm.geometric_parameters.l
delta = pybamm.geometric_parameters.delta

# Tab geometry
l_tab_n = pybamm.geometric_parameters.l_tab_n
centre_y_tab_n = pybamm.geometric_parameters.centre_y_tab_n
centre_z_tab_n = pybamm.geometric_parameters.centre_z_tab_n
l_tab_p = pybamm.geometric_parameters.l_tab_p
centre_y_tab_p = pybamm.geometric_parameters.centre_y_tab_p
centre_z_tab_p = pybamm.geometric_parameters.centre_z_tab_p

# Microscale geometry

inputs = {"Through-cell distance (x_n) [m]": pybamm.standard_spatial_vars.x_n}
epsilon_n = pybamm.FunctionParameter("Negative electrode porosity", inputs)

inputs = {"Through-cell distance (x_s) [m]": pybamm.standard_spatial_vars.x_s}
epsilon_s = pybamm.FunctionParameter("Separator porosity", inputs)

inputs = {"Through-cell distance (x_p) [m]": pybamm.standard_spatial_vars.x_p}
epsilon_p = pybamm.FunctionParameter("Positive electrode porosity", inputs)

epsilon = pybamm.Concatenation(epsilon_n, epsilon_s, epsilon_p)

epsilon_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
epsilon_s_p = pybamm.Parameter("Positive electrode active material volume fraction")
epsilon_inactive_n = 1 - epsilon_n - epsilon_s_n
epsilon_inactive_s = 1 - epsilon_s
epsilon_inactive_p = 1 - epsilon_p - epsilon_s_p
a_n = a_n_dim * R_n
a_p = a_p_dim * R_p

# Electrode Properties
sigma_cn = sigma_cn_dimensional * potential_scale / i_typ / L_x
sigma_n = sigma_n_dim * potential_scale / i_typ / L_x
sigma_p = sigma_p_dim * potential_scale / i_typ / L_x
sigma_cp = sigma_cp_dimensional * potential_scale / i_typ / L_x
sigma_cn_prime = sigma_cn * delta ** 2
sigma_n_prime = sigma_n * delta
sigma_p_prime = sigma_p * delta
sigma_cp_prime = sigma_cp * delta ** 2
sigma_cn_dbl_prime = sigma_cn_prime * delta
sigma_cp_dbl_prime = sigma_cp_prime * delta
# should rename this to avoid confusion with Butler-Volmer
alpha = 1 / (sigma_cn * delta ** 2 * l_cn) + 1 / (sigma_cp * delta ** 2 * l_cp)
alpha_prime = alpha / delta

# Electrolyte Properties


def t_plus(c_e):
    "Dimensionless transference number (i.e. c_e is dimensionless)"
    inputs = {"Electrolyte concentration [mol.m-3]": c_e * c_e_typ}
    return pybamm.FunctionParameter("Cation transference number", inputs)


def one_plus_dlnf_dlnc(c_e):
    inputs = {"Electrolyte concentration [mol.m-3]": c_e * c_e_typ}
    return pybamm.FunctionParameter("1 + dlnf/dlnc", inputs)


beta_surf = pybamm.Scalar(0)


# (1-2*t_plus) is for Nernst-Planck
# 2*(1-t_plus) for Stefan-Maxwell
# Bizeray et al (2016) "Resolving a discrepancy ..."
def chi(c_e):
    return (2 * (1 - t_plus(c_e))) * (one_plus_dlnf_dlnc(c_e))


# Electrochemical Reactions
C_dl_n = (
    C_dl_n_dimensional * potential_scale / interfacial_current_scale_n / tau_discharge
)
C_dl_p = (
    C_dl_p_dimensional * potential_scale / interfacial_current_scale_p / tau_discharge
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

rho_k = pybamm.thermal_parameters.rho_k
rho = rho_n * l_n + rho_s * l_s + rho_p * l_p

lambda_cn = pybamm.thermal_parameters.lambda_cn
lambda_n = pybamm.thermal_parameters.lambda_n
lambda_s = pybamm.thermal_parameters.lambda_s
lambda_p = pybamm.thermal_parameters.lambda_p
lambda_cp = pybamm.thermal_parameters.lambda_cp

lambda_k = pybamm.thermal_parameters.lambda_k

Theta = pybamm.thermal_parameters.Theta

h_edge = pybamm.thermal_parameters.h_edge
h_tab_n = pybamm.thermal_parameters.h_tab_n
h_tab_p = pybamm.thermal_parameters.h_tab_p
h_cn = pybamm.thermal_parameters.h_cn
h_cp = pybamm.thermal_parameters.h_cp
h_total = pybamm.thermal_parameters.h_total

B = (
    i_typ
    * R
    * T_ref
    * tau_th_yz
    / (pybamm.thermal_parameters.rho_eff_dim * F * Delta_T * L_x)
)

T_amb_dim = pybamm.thermal_parameters.T_amb_dim
T_amb = pybamm.thermal_parameters.T_amb

# Initial conditions
T_init = pybamm.thermal_parameters.T_init
c_e_init = c_e_init_dimensional / c_e_typ


def c_n_init(x):
    "Dimensionless initial concentration as a function of dimensionless position x"
    return c_n_init_dimensional(x) / c_n_max


def c_p_init(x):
    "Dimensionless initial concentration as a function of dimensionless position x"
    return c_p_init_dimensional(x) / c_p_max


# --------------------------------------------------------------------------------------
"5. Dimensionless Functions"


def D_e(c_e, T):
    "Dimensionless electrolyte diffusivity"
    c_e_dimensional = c_e * c_e_typ
    T_dim = Delta_T * T + T_ref
    return D_e_dimensional(c_e_dimensional, T_dim) / D_e_typ


def kappa_e(c_e, T):
    "Dimensionless electrolyte conductivity"
    c_e_dimensional = c_e * c_e_typ
    kappa_scale = F ** 2 * D_e_typ * c_e_typ / (R * T_ref)
    T_dim = Delta_T * T + T_ref
    return kappa_e_dimensional(c_e_dimensional, T_dim) / kappa_scale


def D_n(c_s_n, T):
    "Dimensionless negative particle diffusivity"
    sto = c_s_n
    T_dim = Delta_T * T + T_ref
    return D_n_dimensional(sto, T_dim) / D_n_dimensional(pybamm.Scalar(1), T_ref)


def D_p(c_s_p, T):
    "Dimensionless positive particle diffusivity"
    sto = c_s_p
    T_dim = Delta_T * T + T_ref
    return D_p_dimensional(sto, T_dim) / D_p_dimensional(pybamm.Scalar(1), T_ref)


def j0_n(c_e, c_s_surf, T):
    "Dimensionless negative exchange-current density"
    c_e_dim = c_e * c_e_typ
    c_s_surf_dim = c_s_surf * c_n_max
    T_dim = Delta_T * T + T_ref

    return j0_n_dimensional(c_e_dim, c_s_surf_dim, T_dim) / j0_n_ref_dimensional


def j0_p(c_e, c_s_surf, T):
    "Dimensionless positive exchange-current density"
    c_e_dim = c_e * c_e_typ
    c_s_surf_dim = c_s_surf * c_p_max
    T_dim = Delta_T * T + T_ref

    return j0_p_dimensional(c_e_dim, c_s_surf_dim, T_dim) / j0_p_ref_dimensional


def U_n(c_s_n, T):
    "Dimensionless open-circuit potential in the negative electrode"
    sto = c_s_n
    T_dim = Delta_T * T + T_ref
    return (U_n_dimensional(sto, T_dim) - U_n_ref) / potential_scale


def U_p(c_s_p, T):
    "Dimensionless open-circuit potential in the positive electrode"
    sto = c_s_p
    T_dim = Delta_T * T + T_ref
    return (U_p_dimensional(sto, T_dim) - U_p_ref) / potential_scale


def dUdT_n(c_s_n):
    "Dimensionless entropic change in negative open-circuit potential"
    sto = c_s_n
    return dUdT_n_dimensional(sto) * Delta_T / potential_scale


def dUdT_p(c_s_p):
    "Dimensionless entropic change in positive open-circuit potential"
    sto = c_s_p
    return dUdT_p_dimensional(sto) * Delta_T / potential_scale


# --------------------------------------------------------------------------------------
# 6. Input current and voltage

dimensional_current_with_time = pybamm.FunctionParameter(
    "Current function [A]", {"Time [s]": pybamm.t * timescale}
)
dimensional_current_density_with_time = dimensional_current_with_time / (
    n_electrodes_parallel * pybamm.geometric_parameters.A_cc
)
current_with_time = (
    dimensional_current_with_time / I_typ * pybamm.Function(np.sign, I_typ)
)


"Remove any temporary variables"
del inputs
