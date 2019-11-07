#
# Standard thermal parameters
#
import pybamm

# --------------------------------------------------------------------------------------
# Dimensional parameters

# Reference temperature
T_ref = pybamm.Parameter("Reference temperature [K]")

# Density
rho_cn_dim = pybamm.Parameter("Negative current collector density [kg.m-3]")
rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")
rho_s_dim = pybamm.Parameter("Separator density [kg.m-3]")
rho_p_dim = pybamm.Parameter("Positive electrode density [kg.m-3]")
rho_cp_dim = pybamm.Parameter("Positive current collector density [kg.m-3]")

# Specific heat capacity
c_p_cn_dim = pybamm.Parameter(
    "Negative current collector specific heat capacity [J.kg-1.K-1]"
)
c_p_n_dim = pybamm.Parameter("Negative electrode specific heat capacity [J.kg-1.K-1]")
c_p_s_dim = pybamm.Parameter("Separator specific heat capacity [J.kg-1.K-1]")
c_p_p_dim = pybamm.Parameter("Negative electrode specific heat capacity [J.kg-1.K-1]")
c_p_cp_dim = pybamm.Parameter(
    "Positive current collector specific heat capacity [J.kg-1.K-1]"
)

# Thermal conductivity
lambda_cn_dim = pybamm.Parameter(
    "Negative current collector thermal conductivity [W.m-1.K-1]"
)
lambda_n_dim = pybamm.Parameter("Negative electrode thermal conductivity [W.m-1.K-1]")
lambda_s_dim = pybamm.Parameter("Separator thermal conductivity [W.m-1.K-1]")
lambda_p_dim = pybamm.Parameter("Positive electrode thermal conductivity [W.m-1.K-1]")
lambda_cp_dim = pybamm.Parameter(
    "Positive current collector thermal conductivity [W.m-1.K-1]"
)

# Effective thermal properties
rho_eff_dim = (
    rho_cn_dim * c_p_cn_dim * pybamm.geometric_parameters.L_cn
    + rho_n_dim * c_p_n_dim * pybamm.geometric_parameters.L_n
    + rho_s_dim * c_p_s_dim * pybamm.geometric_parameters.L_s
    + rho_p_dim * c_p_p_dim * pybamm.geometric_parameters.L_p
    + rho_cp_dim * c_p_cp_dim * pybamm.geometric_parameters.L_cp
) / pybamm.geometric_parameters.L
lambda_eff_dim = (
    lambda_cn_dim * pybamm.geometric_parameters.L_cn
    + lambda_n_dim * pybamm.geometric_parameters.L_n
    + lambda_s_dim * pybamm.geometric_parameters.L_s
    + lambda_p_dim * pybamm.geometric_parameters.L_p
    + lambda_cp_dim * pybamm.geometric_parameters.L_cp
) / pybamm.geometric_parameters.L

# Cooling coefficient
h_dim = pybamm.Parameter("Heat transfer coefficient [W.m-2.K-1]")

# Typical temperature rise
Phi_dim = pybamm.Scalar(1)  # typical scale for voltage drop across cell (order 1V)
Delta_T = (
    pybamm.electrical_parameters.i_typ * Phi_dim / h_dim
)  # computed from balance of typical cross-cell Ohmic heating with surface heat loss

# Activation energies
E_r_n = pybamm.Parameter("Negative reaction rate activation energy [J.mol-1]")
E_r_p = pybamm.Parameter("Positive reaction rate activation energy [J.mol-1]")
E_D_s_n = pybamm.Parameter("Negative solid diffusion activation energy [J.mol-1]")
E_D_s_p = pybamm.Parameter("Positive solid diffusion activation energy [J.mol-1]")
E_D_e = pybamm.Parameter("Electrolyte diffusion activation energy [J.mol-1]")
E_k_e = pybamm.Parameter("Electrolyte conductivity activation energy [J.mol-1]")

# Initial temperature
T_init_dim = pybamm.Parameter("Initial temperature [K]")

# --------------------------------------------------------------------------------------
# Timescales
tau_th_yz = rho_eff_dim * (pybamm.geometric_parameters.L_z ** 2) / lambda_eff_dim

# --------------------------------------------------------------------------------------
# Dimensionless parameters

rho_cn = rho_cn_dim * c_p_cn_dim / rho_eff_dim
rho_n = rho_n_dim * c_p_n_dim / rho_eff_dim
rho_s = rho_s_dim * c_p_s_dim / rho_eff_dim
rho_p = rho_p_dim * c_p_p_dim / rho_eff_dim
rho_cp = rho_cp_dim * c_p_cp_dim / rho_eff_dim

rho_k = pybamm.Concatenation(
    pybamm.FullBroadcast(rho_n, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(rho_s, ["separator"], "current collector"),
    pybamm.FullBroadcast(rho_p, ["positive electrode"], "current collector"),
)

lambda_cn = lambda_cn_dim / lambda_eff_dim
lambda_n = lambda_n_dim / lambda_eff_dim
lambda_s = lambda_s_dim / lambda_eff_dim
lambda_p = lambda_p_dim / lambda_eff_dim
lambda_cp = lambda_cp_dim / lambda_eff_dim

lambda_k = pybamm.Concatenation(
    pybamm.FullBroadcast(lambda_n, ["negative electrode"], "current collector"),
    pybamm.FullBroadcast(lambda_s, ["separator"], "current collector"),
    pybamm.FullBroadcast(lambda_p, ["positive electrode"], "current collector"),
)


Theta = Delta_T / T_ref
h = h_dim * pybamm.geometric_parameters.L_x / lambda_eff_dim

T_init = (T_init_dim - T_ref) / Delta_T
