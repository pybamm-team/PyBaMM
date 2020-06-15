#
# Base class for particle cracking model
# It simulates how much surface area is created by cracks during battery cycling
# For setting up now and to be finished later

import pybamm
    from .base_cracking import BaseCracking

class ParticleCracking(BaseCracking):
    """cracking behaviour in electrode particles.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    requiring the radius, average concantration, surface concantration
    """
    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        l_cr_n=pybamm.standard_variables.l_cr_n # crack length in anode particles
        # L_sei_cr_n=
        # L_plating_cr_n=
        return variables
        

    def get_coupled_variables(self, variables):
        l_cr_n=variables["Negative electrode crack length"]
        variables.update(self._get_standard_surface_variables(l_cr_n))
        return variables

    def set_rhs(self,variables)
        T_n=variables["Negative electrode temperature"]
        l_cr_n=variables["Negative electrode crack length"]  
        stress_t_surf_n=variables["Negative electrode surface tangential stress"]
        mp= pybamm.mechanical_parameters
        R = pybamm.standard_parameters_lithium_ion.R
        Delta_T = pybamm.thermal_parameters.Delta_T
        k_cr_n=mp.k_cr*exp( mp.Eac_cr/R*(1/T_n/Delta_T-1/mp.T_ref) ) # cracking rate with temperature dependence
        stress_t_surf_n(stress_t_surf_n<0)=0 # compressive stress will not lead to crack propagation
        dK_SIF = stress_t_surf_n*mp.b_cr*sqrt(pi*l_cr_n)
        dl_cr_n=mp.crack_flag*k_cr_n*dK_SIF^mp.m_cr/mp.t0_cr
        self.rhs={l_cr_n: dl_cr_n}

    def set_initial_conditions(self,variables):
        l_cr_n=variables["Negative electrode crack length"]
        l_cr_n_0=pybamm.mechanical_parameters.l_cr_n_0

        self.initial_conditions={l_cr_n:l_cr_n_0}

    # same code for the cathode with changing "_p" to "_n"
    #