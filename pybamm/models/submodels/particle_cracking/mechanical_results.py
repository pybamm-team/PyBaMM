#
# Base class for particle cracking model
# It calculate the stress and displacements in electrode particles
# For setting up now and to be finished later

import pybamm

class MechanicalResults(pybamm.BaseSubModel):
    """Base class for mechanical results of stresses and displacement.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms
    **Extends:** :class:`pybamm.BaseSubModel`
    """
    def __init__(self, param, domain):
        self.domain = domain
        super().__init__(param)

    def get_mechanical_results(self,domain,variables):
        """
        calculate the mechanical results of stresses and displacement
        Parameters
        ----------
        Returns
        -------
        variables : dict
        The variables of radial and tangential stresses and surface displacement
        """
        c_s_n=variables["Negative particle concentration"] 
        c_s_n_avg=pybamm.r_average(c_s_n)
        # need to check whether is avarage cs in a particle
        c_s_n_surf=variables["Negative particle surface concentration"]
        mp=pybamm.mechanical_parameters
        disp_n_surf_dim=mp.Omega_n*mp.R_n/3*(c_s_n_avg-mp.c_n_0) # c0 reference concentration for no deformation
        stress_r_n_surf_dim=0*mp.E_n
        stress_t_n_surf_dim=mp.Omega_n*mp.E_n/3.0/(1.0-mp.nu_n) *(c_s_n_avg - c_s_n_surf)

        disp_n_surf=disp_n_surf_dim/mp.R_n
        stress_r_n_surf=stress_r_n_surf_dim/mp.E_n
        stress_t_n_surf=stress_t_n_surf_dim/mp.E_n
        # stress_r_n_centre=2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)(c_s_n_avg-Cs_n_centre)
        # stress_t_n_centre=2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)*(c_s_n_avg-Cs_n_centre)

        mp_results = {
            "Negative particle surface tangential stress": stress_t_n_surf,
            "Negative particle surface radial stress": stress_r_n_surf,
            "Negative particle surface displacement": disp_n_surf,
            "Negative particle surface tangential stress [Pa]": stress_t_n_surf_dim,
            "Negative particle surface radial stress [Pa]": stress_r_n_surf_dim,
            "Negative particle surface displacement [m]": disp_n_surf_dim,
        }
        variables.update(mp_results)
        return variables
    
        # same code for cathode with replacing "_n" with "_p"
        