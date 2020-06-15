#
# Base class for particle cracking model
# It calculate the stress and displacements in electrode particles
# For setting up now and to be finished later

import pybamm

class MechanicalResults(BaseModel):
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

    def _get_mechanical_results(self)
    """
    calculate the mechanical results of stresses and displacement
    Parameters
    ----------
    Returns
    -------
    variables : dict
        The variables of radial and tangential stresses and surface displacement
    """
        c_s_n_avg=variables["Negative particle average concentration"] 
        # need to check whether is avarage cs in a particle
        c_s_n_surf=variables["Negative particle surface concentration"]
        mp=pybamm.mechanical_parameters
        disp_surf_n=mp.Omega_n*mp.R_n/3*(c_s_n_avg-mp.c_n_0) # c0 reference concentration for no deformation
        stress_r_surf_n=0
        stress_t_surf_n=mp.Omega_n*mp.E_n/3.0/(1.0-mp.nu_n) *(c_s_n_avg - c_s_n_surf)
        # stress_r_centre_n=2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)(c_s_n_avg-Cs_n_centre)
        # stress_t_centre_n=2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)*(c_s_n_avg-Cs_n_centre)

        variables = {
            "Negative particle surface tangentrial stress [Pa]": stress_t_surf_n,
            "Negative particle surface radial stress [Pa]": stress_r_surf_n,
            "Negative particle surface displacement [m]": disp_surf_n,
        }
        return variables
    
        # same code for cathode with replacing "_n" with "_p"
