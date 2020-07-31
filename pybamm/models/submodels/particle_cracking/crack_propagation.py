#
# Base class for particle cracking model
# It simulates how much surface area is created by cracks during battery cycling
# For setting up now and to be finished later

import pybamm
from .base_cracking import BaseCracking
import numpy as np

class CrackPropagation(BaseCracking):
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
        return self.get_standard_variables()
        
    def get_coupled_variables(self, variables):
        variables.update(self._get_mechanical_results(variables))
        return variables

    def set_rhs(self,variables):
        T_n = variables["Negative electrode temperature"]
        stress_t_surf_n = variables["Negative particle surface tangential stress [Pa]"]
        l_cr_n = variables["Negative particle crack length"]
        # crack length in anode particles
        mp = pybamm.mechanical_parameters
        R = pybamm.standard_parameters_lithium_ion.R
        Delta_T = pybamm.thermal_parameters.Delta_T
        l_cr_n_0 = pybamm.mechanical_parameters.l_cr_n_0
        k_cr_n=mp.k_cr * pybamm.exp( mp.Eac_cr / R * (1 / T_n / Delta_T - 1 / mp.T_ref)) 
        # cracking rate with temperature dependence
        # stress_t_surf_n[stress_t_surf_n<0]=pybamm.Scalr(0) 
        # # compressive stress will not lead to crack propagation
        dK_SIF = stress_t_surf_n * mp.b_cr * pybamm.Sqrt(np.pi * l_cr_n * l_cr_n_0) * (stress_t_surf_n >= 0)
        dl_cr_n = mp.crack_flag * k_cr_n * pybamm.Power(dK_SIF , mp.m_cr) / mp.t0_cr / l_cr_n_0
        self.rhs = {l_cr_n: dl_cr_n}

    def set_initial_conditions(self,variables):
        l_cr_n = variables["Negative particle crack length"]
        l0 = pybamm.PrimaryBroadcast(
            pybamm.Scalar(1), ["negative electrode"]
        )
        self.initial_conditions={l_cr_n: l0}

    # same code for the cathode with changing "_p" to "_n"
    #