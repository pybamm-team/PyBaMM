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
            self.domain + " particle crack length",
            domain=self.domain.lower() + " electrode",
        )
        # crack length in anode particles
        l_cr_n_dim = pybamm.Variable(
            self.domain + " particle crack length [m]",
            domain=self.domain.lower() + " electrode",
        )
        domain = self.domain.lower() + " particle"
        L_cr_n_av = pybamm.Variable(
            f"X-averaged {domain} crack length [m]", 
            domain="current collector"
        )
        if self.domain == "Positive":
            l_cr_n0 = pybamm.mechanical_parameters.l_cr_p_0
        else:
            l_cr_n0 = pybamm.mechanical_parameters.l_cr_n_0
        l_cr_n_av = pybamm.x_average(l_cr_n)
        # crack length [m] in anode particles
        variables = {
            self.domain + " particle crack length [m]": l_cr_n * l_cr_n0,
            self.domain + " particle crack length": l_cr_n,
            f"X-averaged {domain} crack length": l_cr_n_av,
            f"X-averaged {domain} crack length [m]": l_cr_n_av * l_cr_n0,
        }
        variables.update(self._get_standard_surface_variables(l_cr_n))
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
        c_s_n = variables[self.domain + " particle concentration"]
        c_s_n_avg = pybamm.r_average(c_s_n)  # average concentration for particles
        # c_s_n_avg = variables["R-average " + self.domain.lower() + " particle concentration"]
        c_s_n_surf = variables[self.domain + " particle surface concentration"]
        # c_s_n_avg = 2*c_s_n_surf
        mp = pybamm.mechanical_parameters

        if self.domain == "Negative":
            Omega_n = mp.Omega_n
            R_n = mp.R_n
            c_scale = self.param.c_n_max
            c_n_0 = mp.c_n_0
            E_n = mp.E_n
            nu_n = mp.nu_n
        elif self.domain == "Positive":
            Omega_n = mp.Omega_p
            R_n = mp.R_p
            c_scale = self.param.c_p_max
            c_n_0 = mp.c_p_0
            E_n = mp.E_p
            nu_n = mp.nu_p

        disp_n_surf_dim = Omega_n * R_n / 3 * (c_s_n_avg - c_n_0) * c_scale
        # c0 reference concentration for no deformation
        stress_r_n_surf_dim = 0 * E_n
        stress_t_n_surf_dim = (
            Omega_n * E_n / 3.0 / (1.0 - nu_n) * (c_s_n_avg - c_s_n_surf) * c_scale
        )  # noqa
        disp_n_surf = disp_n_surf_dim / R_n
        stress_r_n_surf = stress_r_n_surf_dim / E_n
        stress_t_n_surf = stress_t_n_surf_dim / E_n
        # stress_r_n_centre = 2.0 * Omega_n * E_n / 9.0 / (1.0 - nu_n) * (c_s_n_avg - Cs_n_centre) # noqa
        # stress_t_n_centre = 2.0 * Omega_n * E_n / 9.0 / (1.0 - nu_n) * (c_s_n_avg - Cs_n_centre) # noqa

        return {
            self.domain + " particle surface tangential stress": stress_t_n_surf,
            self.domain + " particle surface radial stress": stress_r_n_surf,
            self.domain + " particle surface displacement": disp_n_surf,
            self.domain
            + " particle surface tangential stress [Pa]": stress_t_n_surf_dim,
            self.domain + " particle surface radial stress [Pa]": stress_r_n_surf_dim,
            self.domain + " particle surface displacement [m]": disp_n_surf_dim,
        }

    def _get_standard_surface_variables(self, l_cr_n):
        """
        A private function to obtain the standard variables which
        can be derived from the local particle crack surfaces.
        Parameters
        ----------
        l_cr_n : :class:`pybamm.Symbol`
            The crack length in electrode particles.
        Returns
        -------
        variables : dict
        The variables which can be derived from the crack length.
        """
        rho_cr = pybamm.mechanical_parameters.rho_cr
        if self.domain == "Negative":
            a_n = pybamm.LithiumIonParameters().a_n
            R_n = pybamm.LithiumIonParameters().R_n
        elif self.domain == "Positive":
            a_n = pybamm.LithiumIonParameters().a_p
            R_n = pybamm.LithiumIonParameters().R_p
        roughness =  l_cr_n * 2 * rho_cr + 1 # the ratio of cracks to normal surface
        a_n_cr = (roughness - 1) * a_n # normalised crack surface area
        a_n_cr_dim = a_n_cr / R_n  # crack surface area to volume ratio [m-1]
        # a_n_cr_xavg=pybamm.x_average(a_n_cr)
        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            self.domain + " crack surface to volume ratio [m-1]": a_n_cr_dim,
            self.domain + " crack surface to volume ratio": a_n_cr,
            # self.domain + " X-averaged crack surface to volume ratio [m-1]": a_n_cr_xavg / R_n,
            # self.domain + " X-averaged crack surface to volume ratio": a_n_cr_xavg,
            self.domain + " electrode roughness ratio": roughness,
            f"X-averaged {self.domain.lower()} electrode roughness ratio": roughness_xavg,
        }
        return variables
