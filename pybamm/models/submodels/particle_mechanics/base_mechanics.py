#
# Base class for particle cracking models.
#
import pybamm


class BaseCracking(pybamm.BaseSubModel):
    """
    Base class for particle cracking models. See [1]_ for mechanical model (thickness
    change) and [2]_ for cracking model.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : dict, optional
        Dictionary of either the electrode for "Positive" or "Nagative"

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2019). Electrochemical
           Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch
           Cells. Journal of The Electrochemical Society, 167(1), 013512.
    .. [2] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
           Battery cycle life prediction with coupled chemical degradation and
           fatigue mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

        pybamm.citations.register("Ai2019")
        pybamm.citations.register("Deshpande2012")

    def _get_standard_variables(self, l_cr):
        domain = self.domain.lower() + " particle"
        if self.domain == "Positive":
            l_cr0 = self.param.l_cr_p_0
        else:
            l_cr0 = self.param.l_cr_n_0
        l_cr_av = pybamm.x_average(l_cr)
        variables = {
            self.domain + " particle crack length [m]": l_cr * l_cr0,
            self.domain + " particle crack length": l_cr,
            f"X-averaged {domain} crack length": l_cr_av,
            f"X-averaged {domain} crack length [m]": l_cr_av * l_cr0,
        }
        return variables

    def _get_mechanical_results(self, variables):
        c_s_rav = variables[
            "R-averaged " + self.domain.lower() + " particle concentration"
        ]
        c_s_surf = variables[self.domain + " particle surface concentration"]
        T_xav = variables["X-averaged cell temperature"]
        eps_s = variables[self.domain + " electrode active material volume fraction"]

        if "Cell thickness change [m]" not in variables:
            cell_thickness_change = (
                T_xav * self.param.Delta_T * self.param.alpha_T_cell_dim
            )  # thermal expansion
        else:
            cell_thickness_change = variables["Cell thickness change [m]"]

        if self.domain == "Negative":
            x = pybamm.standard_spatial_vars.x_n
            Omega = self.param.Omega_n
            R0 = self.param.R_n(x)
            c_scale = self.param.c_n_max
            c_0 = self.param.c_n_0
            E0 = self.param.E_n
            nu = self.param.nu_n
            L0 = self.param.L_n
            c_init = self.param.c_n_init(1)
            v_change = pybamm.x_average(
                eps_s * self.param.t_n_change(c_s_rav)
            ) - pybamm.x_average(eps_s * self.param.t_n_change(c_init))

        elif self.domain == "Positive":
            x = pybamm.standard_spatial_vars.x_p
            Omega = self.param.Omega_p
            R0 = self.param.R_p(x)
            c_scale = self.param.c_p_max
            c_0 = self.param.c_p_0
            E0 = self.param.E_p
            nu = self.param.nu_p
            L0 = self.param.L_p
            c_init = self.param.c_p_init(0)
            v_change = pybamm.x_average(
                eps_s * self.param.t_p_change(c_s_rav)
            ) - pybamm.x_average(eps_s * self.param.t_p_change(c_init))

        cell_thickness_change += self.param.n_electrodes_parallel * v_change * L0
        disp_surf_dim = Omega * R0 / 3 * (c_s_rav - c_0) * c_scale
        # c0 reference concentration for no deformation
        stress_r_surf_dim = 0 * E0
        stress_t_surf_dim = (
            Omega * E0 / 3.0 / (1.0 - nu) * (c_s_rav - c_s_surf) * c_scale
        )
        disp_surf = disp_surf_dim / R0
        stress_r_surf = stress_r_surf_dim / E0
        stress_t_surf = stress_t_surf_dim / E0
        stress_r_surf_av = pybamm.x_average(stress_r_surf)
        stress_t_surf_av = pybamm.x_average(stress_t_surf)

        return {
            self.domain + " particle surface tangential stress": stress_t_surf,
            self.domain + " particle surface radial stress": stress_r_surf,
            self.domain + " particle surface displacement": disp_surf,
            self.domain + " particle surface tangential stress [Pa]": stress_t_surf_dim,
            self.domain + " particle surface radial stress [Pa]": stress_r_surf_dim,
            self.domain + " particle surface displacement [m]": disp_surf_dim,
            "X-averaged "
            + self.domain.lower()
            + " particle surface radial stress": stress_r_surf_av,
            "X-averaged "
            + self.domain.lower()
            + " particle surface radial stress [Pa]": stress_r_surf_av * E0,
            "X-averaged "
            + self.domain.lower()
            + " particle surface tangential stress": stress_t_surf_av,
            "X-averaged "
            + self.domain.lower()
            + " particle surface tangential stress [Pa]": stress_t_surf_av * E0,
            "Cell thickness change [m]": cell_thickness_change,
        }

    def _get_standard_surface_variables(self, variables):
        """
        A private function to obtain the standard variables which
        can be derived from the local particle crack surfaces.

        Parameters
        ----------
        l_cr : :class:`pybamm.Symbol`
            The crack length in electrode particles.
        a0 : :class:`pybamm.Symbol`
            Smooth surface area to volume ratio.

        Returns
        -------
        variables : dict
            The variables which can be derived from the crack length.
        """
        l_cr = variables[self.domain + " particle crack length"]
        a0 = variables[self.domain + " electrode surface area to volume ratio"]
        if self.domain == "Negative":
            x = pybamm.standard_spatial_vars.x_n
            R0 = self.param.R_n(x)
            rho_cr = self.param.rho_cr_n
        elif self.domain == "Positive":
            x = pybamm.standard_spatial_vars.x_p
            R0 = self.param.R_p(x)
            rho_cr = self.param.rho_cr_p
        roughness = l_cr * 2 * rho_cr + 1  # the ratio of cracks to normal surface
        a_cr = (roughness - 1) * a0  # normalised crack surface area
        a_cr_dim = a_cr / R0  # crack surface area to volume ratio [m-1]

        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            self.domain + " crack surface to volume ratio [m-1]": a_cr_dim,
            self.domain + " crack surface to volume ratio": a_cr,
            self.domain + " electrode roughness ratio": roughness,
            f"X-averaged {self.domain.lower()} "
            "electrode roughness ratio": roughness_xavg,
        }
        return variables
