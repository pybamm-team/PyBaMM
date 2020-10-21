#
# Standard parameters for mechanical models
#

import pybamm

# --------------------------------------------------------------------------------------
# Should find a proper place to add the following mechanics parameters
"""
# values are from W.Ai et al. Journal of The Electrochemical Society,
# 2020 167 013512 [DOI: 10.1149/2.0122001JES]
# and J. Purewal et al. Journal of Power Sources 272 (2014) 1154-1161
"""
# Dimensional parameters

flag_mechanics = pybamm.Scalar(1)  # flag: 1 - activate mechanical effects; 0 - disable
nu_p = pybamm.Parameter("Positive electrode Poisson's ratio")
E_p = pybamm.Parameter("Positive electrode Young's modulus [Pa]")
c_p_0 = pybamm.Parameter(
    "Positive electrode Reference concentration for free of deformation [m3.mol-1]"
)  # noqa
Omega_p = pybamm.Parameter("Positive electrode Partial molar volume [m3.mol-1]")
nu_n = pybamm.Parameter("Negative electrode Poisson's ratio")
E_n = pybamm.Parameter("Negative electrode Young's modulus [Pa]")
c_n_0 = pybamm.Parameter(
    "Negative electrode Reference concentration for free of deformation [m3.mol-1]"
)  # noqa
Omega_n = pybamm.Parameter("Negative electrode Partial molar volume [m3.mol-1]")

flag_crack = pybamm.Scalar(1)
#  1 - enable crack propagation; 0 - disable cracking in battery degradation
l_cr_p_0 = pybamm.Parameter("Positive electrode Initial crack length [m]")
l_cr_n_0 = pybamm.Parameter("Negative electrode Initial crack length [m]")
w_cr = pybamm.Parameter("Negative electrode Initial crack width [m]")
rho_cr_dim = pybamm.Parameter(
    "Negative electrode Number of cracks per unit area of the particle [m-2]"
)  # noqa
b_cr = pybamm.Parameter("Negative electrode Paris' law constant b")
m_cr = pybamm.Parameter("Negative electrode Paris' law constant m")
k_cr = pybamm.Parameter("Negative electrode Cracking rate")
Eac_cr = pybamm.Parameter(
    "Negative electrode Activation energy for cracking rate [kJ.mol-1]"
)  # noqa
alpha_T_cell_dim = pybamm.Parameter("Cell thermal expansion coefficien [m.K-1]")
n_layers = pybamm.Parameter("Number of electrodes connected in parallel to make a cell")

T_ref = pybamm.LithiumIonParameters().T_ref  # [K]
R_n = pybamm.LithiumIonParameters().R_n # [m]
R_p = pybamm.LithiumIonParameters().R_p # [m]
R_const = pybamm.constants.R
c_p_max = pybamm.LithiumIonParameters().c_p_max
c_n_max = pybamm.LithiumIonParameters().c_n_max
T_ref = pybamm.LithiumIonParameters().T_ref

theta_p_dim = flag_mechanics * Omega_p ** 2 / R_const * 2 / 9 * E_p / (1 - nu_p)
# intermediate variable  [K*m^3/mol]
theta_n_dim = flag_mechanics * Omega_n ** 2 / R_const * 2 / 9 * E_n / (1 - nu_n)
# intermediate variable  [K*m^3/mol]

# Dimensionless parameters
rho_cr = rho_cr_dim * l_cr_n_0 * w_cr
theta_p = theta_p_dim * c_p_max / T_ref
theta_n = theta_n_dim * c_n_max / T_ref
t0_cr = 3600/pybamm.LithiumIonParameters().timescale # nomarlised typical time for one cycle


