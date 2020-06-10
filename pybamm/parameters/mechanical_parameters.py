#
# Standard parameters for mechanical models
#

import pybamm

# --------------------------------------------------------------------------------------
# Should find a proper place to add the following mechanics parameters
'''
# values are from W.Ai et al. Journal of The Electrochemical Society, 2020 167 013512 [DOI: 10.1149/2.0122001JES]
# and J. Purewal et al. Journal of Power Sources 272 (2014) 1154-1161

param.mechanics=1   # flag: 1 - activate mechanical effects; 0 - ignore
param.nu_p=0.2  # Poisson's ratio for cathode
param.E_p=375e9 # Young's modulus for cathode [GPa]
param.c_p_0 # Reference concentration when cathode particle is free of stress 
param.Omega_p= -7.28e-7 # Partial molar volume for cathode, Dimentionless 
param.theta_p= param.Omega_p**2/param.R*2/9*param.E_p*(1-param.nu_p) # intermediate variable  [K*m^3/mol]
param.nu_n=0.3  # Poisson's ratio for anode
param.E_n=15e9 # Young's modulus for anode [GPa]
param.c_n_0 # Reference concentration when anode particle is free of stress, Dimentionless 
param.Omega_n= 3.1e-6 # Partial molar volume for anode [m^3/mol]
param.theta_n= param.Omega_n**2/param.R*2/9*param.E_n*(1-param.nu_n) # intermediate variable  [K*m^3/mol]

# Below parameters from JES 159(10) A1730-A1738 (2012)
param.crackmodel=0; %  1 - enable crack propagation; 0 - disable cracking in battery degradation 
param.l_cr0_p=20e-9; % initial crack length (m)
param.l_cr0_n=20e-9;
param.w_cr=15e-9; % initial crack width (m)
param.rho_cr=3.18e15; % number of cracks per unit area of the particle (m^-2)
param.b_cr=1.12; % Paris' law constant
param.m_cr=2.2; % Paris' law constant
param.k_cr=1.9e-9; % Arrhenius constant for Paris' law  %1.62e-16;

'''
# Dimensional parameters

mechanics_flag=pybamm.Parameter("Mechanics flag")   # flag: 1 - activate mechanical effects; 0 - disable
nu_p=pybamm.Parameter("Poisson's ratio for cathode")
E_p=pybamm.Parameter("Young's modulus for cathode [GPa]")
c_p_0=pybamm.Parameter("Reference concentration when cathode particle is free of stress")
Omega_p=pybamm.Parameter("Partial molar volume for cathode [m^3/mol]")
nu_n=pybamm.Parameter("Poisson's ratio for anode")
E_n=pybamm.Parameter("Young's modulus for anode [GPa]")
c_n_0=pybamm.Parameter("Reference concentration when anode particle is free of stress")
Omega_n=pybamm.Parameter("Partial molar volume for anode [m^3/mol]")

crack_flag=pybamm.Parameter("Crack flag")  #  1 - enable crack propagation; 0 - disable cracking in battery degradation 
l_cr0_p=pybamm.Parameter("initial crack length in cathode particles [m]")
l_cr0_n=pybamm.Parameter("initial crack length in anode particles [m]")
w_cr=pybamm.Parameter("initial crack width [m]")
rho_cr=pybamm.Parameter("number of cracks per unit area of the particle [m^-2]")
b_cr=pybamm.Parameter("Paris' law constant")
m_cr=pybamm.Parameter("Paris' law constant")
k_cr=pybamm.Parameter("Paris' law constant for cracking rate")
Eac_cr=pybamm.Parameter("Activation energy for cracking rate")

theta_p= Omega_p**2/param.Radius*2/9*E_p*(1-nu_p) # intermediate variable  [K*m^3/mol]
theta_n= Omega_n**2/param.Radius*2/9*E_n*(1-nu_n) # intermediate variable  [K*m^3/mol]

# Dimensionless parameters