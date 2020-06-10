#
# Base class for particle cracking model
# It simulates how much surface area is created by cracks during battery cycling
# For setting up now and to be finished later

import pybamm


class ParticleCracking(pybamm.BaseSubModel):
    """cracking behaviour in electrode particles.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

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
# parameters from JES 159(10) A1730-A1738 (2012)
param.crackmodel=0; %  1 - enable crack propagation; 0 - disable cracking in battery degradation 
param.l_cr0_p=20e-9; % initial crack length (m)
param.l_cr0_n=20e-9;
param.w_cr=15e-9; % initial crack width (m)
param.rho_cr=3.18e15; % number of cracks per unit area of the particle (m^-2)
param.b_cr=1.12; % Paris' law constant
param.m_cr=2.2; % Paris' law constant
param.k_cr=1.9e-9; % cracking rate constant for Paris' law  %1.62e-16;
param.Eac_cr=8000; % Arrhenius constant for Paris' law  %1.62e-16;

requiring the radius, average concantration, surface concantration
    """
    # for the anode
    disp_surf_n=Omega_n*R_n/3*(Cs_n_avg-C0) # c0 reference concentration for no deformation
    stress_r_surf_n=0
    stress_t_surf_n=Omega_n*E_n/3.0/(1.0-nu_n) *(Cs_n_avg - Cs_n_surf)
    stress_r_centre_n=2.0*Omega_n*E_n/9.0/(1.0-nu_n)(Cs_n_avg-Cs_n_centre)
    stress_t_centre_n=2.0*Omega_n*E_n/9.0/(1.0-nu_n)*(Cs_n_avg-Cs_n_centre)

    k_cr_n=k_cr*exp( Eac_cr/R*(1./T-1/Tref) )
    stress(stress_t_surf_n<0)=0; # compressive stress will not lead to crack propagation
    dK_SIF = stress.*b_cr.*sqrt(pi*l_cr)
    dl_cr=k_cr.*dK_SIF.^m_cr/t0_cr
    da_n_cr= a_n*dl_cr*2*w_cr*rho_cr; # crack surface area

    # same code for the cathode with changing "_p" to "_n"
    #