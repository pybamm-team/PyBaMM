#
# Base class for particle cracking models.
#
import pybamm


class BaseCracking(pybamm.BaseSubModel):
    """Base class for particle cracking models.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : dict, optional
        Dictionary of either the electrode for "Positive" or "Nagative"
    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        self.domain = domain
        super().__init__(param)

    def get_standard_variables(self):
        l_cr_n = pybamm.Variable(
            "Negative particle crack length", domain="negative electrode"
        )
        # crack length in anode particles
        l_cr_n_dim = pybamm.Variable(
            "Negative particle crack length", domain="negative electrode"
        )
        # crack length in anode particles
        variables = {
            "Negative particle crack length [m]": l_cr_n_dim,
            "Negative particle crack length": l_cr_n,
        }
        variables.update(self._get_standard_surface_variables(l_cr_n))
        return variables

    def _get_standard_surface_variables(self, l_cr_n):
        """
        A private function to obtain the standard variables which
        can be derived from the local particle crack surfaces.
        Parameters
        ----------
        l_cr_n : :class:`pybamm.Symbol`
            The crack length in anode particles.
        Returns
        -------
        variables : dict
        The variables which can be derived from the crack length.
        """
        rho_cr = pybamm.mechanical_parameters.rho_cr
        a_n = pybamm.LithiumIonParameters().a_n
        a_n_cr = l_cr_n * 2 * rho_cr  # crack surface area normalised by a_n
        a_n_cr_dim = a_n_cr * a_n  # crack surface area [m-1]
        # a_n_cr_xavg=pybamm.x_average(a_n_cr)
        variables = {
            "Crack surface to volume ratio [m-1]": a_n_cr_dim,
            "Crack surface to volume ratio": a_n_cr,
            # "X-averaged crack surface to volume ratio [m-1]": a_n_cr_xavg*a_n,
            # "X-averaged crack surface to volume ratio": a_n_cr_xavg,
        }
        return variables

    def _get_mechanical_results(self, variables):
        """
        calculate the mechanical results of stresses and displacement in the anode
        Parameters
        ----------
        Returns
        -------
        variables : dict
        The variables of radial and tangential stresses and surface displacement
        """
        c_s_n = variables["Negative particle concentration"]
        c_s_n_avg = pybamm.r_average(c_s_n)  # average concentration for particles
        # c_s_n_avg = variables["R-average negative particle concentration"]
        # need to check whether is avarage cs in a particle
        c_s_n_surf = variables["Negative particle surface concentration"]
        # c_s_n_avg = 2*c_s_n_surf
        mp = pybamm.mechanical_parameters
        c_scale = self.param.c_n_max
        disp_n_surf_dim = mp.Omega_n * mp.R_n / 3 * (c_s_n_avg - mp.c_n_0) * c_scale
        # c0 reference concentration for no deformation
        stress_r_n_surf_dim = 0 * mp.E_n
        stress_t_n_surf_dim = ( mp.Omega_n * mp.E_n / 3.0 / (1.0 - mp.nu_n) * (c_s_n_avg - c_s_n_surf) * c_scale ) # noqa        
        disp_n_surf = disp_n_surf_dim / mp.R_n
        stress_r_n_surf = stress_r_n_surf_dim / mp.E_n
        stress_t_n_surf = stress_t_n_surf_dim / mp.E_n
        # stress_r_n_centre = 2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)(c_s_n_avg-Cs_n_centre) # noqa
        # stress_t_n_centre = 2.0*mp.Omega_n*mp.E_n/9.0/(1.0-mp.nu_n)*(c_s_n_avg-Cs_n_centre) # noqa

        return {
            "Negative particle surface tangential stress": stress_t_n_surf,
            "Negative particle surface radial stress": stress_r_n_surf,
            "Negative particle surface displacement": disp_n_surf,
            "Negative particle surface tangential stress [Pa]": stress_t_n_surf_dim,
            "Negative particle surface radial stress [Pa]": stress_r_n_surf_dim,
            "Negative particle surface displacement [m]": disp_n_surf_dim,
        }

        # same code for cathode with replacing "_n" with "_p"
