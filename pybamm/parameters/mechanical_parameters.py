#
# Standard parameters for mechanical models
#

import pybamm

# --------------------------------------------------------------------------------------
# Should find a proper place to add the following mechanics parameters
'''
# values are from W.Ai et al. Journal of The Electrochemical Society, 
# 2020 167 013512 [DOI: 10.1149/2.0122001JES]
# and J. Purewal et al. Journal of Power Sources 272 (2014) 1154-1161
'''
# Dimensional parameters

mechanics_flag = 1   # flag: 1 - activate mechanical effects; 0 - disable
nu_p = pybamm.Parameter("Positive electrode Poisson's ratio")
E_p = pybamm.Parameter("Positive electrode Young's modulus [Pa]")
c_p_0 = pybamm.Parameter("Positive electrode Reference concentration " \
                        "for free of deformation [m3.mol-1]")
Omega_p = pybamm.Parameter("Positive electrode Partial molar volume [m3.mol-1]")
nu_n = pybamm.Parameter("Negative electrode Poisson's ratio")
E_n = pybamm.Parameter("Negative electrode Young's modulus [Pa]")
c_n_0 = pybamm.Parameter("Negative electrode Reference concentration " \
                        "for free of deformation [m3.mol-1]")
Omega_n = pybamm.Parameter("Negative electrode Partial molar volume [m3.mol-1]")

crack_flag = 1  
#  1 - enable crack propagation; 0 - disable cracking in battery degradation 
l_cr_p_0 = pybamm.Parameter("Positive electrode Initial crack length [m]")
l_cr_n_0 = pybamm.Parameter("Negative electrode Initial crack length [m]")
w_cr = pybamm.Parameter("Negative electrode Initial crack width [m]")
rho_cr_dim = pybamm.Parameter("Negative electrode Number of cracks " \
                            "per unit area of the particle [m-2]")
b_cr = pybamm.Parameter("Negative electrode Paris' law constant b")
m_cr = pybamm.Parameter("Negative electrode Paris' law constant m")
k_cr = pybamm.Parameter("Negative electrode Cracking rate")
Eac_cr = pybamm.Parameter("Negative electrode Activation energy " \
                            "for cracking rate [kJ.mol-1]")

T_ref = pybamm.standard_parameters_lithium_ion.T_ref # [K]
R_p = pybamm.standard_parameters_lithium_ion.R_p # [m]
R_n = pybamm.standard_parameters_lithium_ion.R_n # [m]
t0_cr = 3600 # typical time for one cycle [s]

theta_p_dim = Omega_p**2 / R_p * 2 / 9 * E_p * (1 - nu_p) 
# intermediate variable  [K*m^3/mol]
theta_n_dim = Omega_n**2 / R_n * 2 / 9 * E_n * (1 - nu_n) 
# intermediate variable  [K*m^3/mol]

# Dimensionless parameters
rho_cr = rho_cr_dim * l_cr_n_0 * w_cr