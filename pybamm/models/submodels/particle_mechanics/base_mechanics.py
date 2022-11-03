#
# Base class for particle cracking models.
#
import pybamm


class BaseMechanics(pybamm.BaseSubModel):
    """
    Base class for particle mechanics models. See [1]_ for mechanical model (thickness
    change) and [2]_ for cracking model.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : dict, optional
        Dictionary of either the electrode for "positive" or "Nagative"
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

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

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)

        pybamm.citations.register("Ai2019")
        pybamm.citations.register("Deshpande2012")

    def _get_standard_variables(self, l_cr):
        domain, Domain = self.domain_Domain
        l_cr_av = pybamm.x_average(l_cr)
        variables = {
            f"{Domain} particle crack length [m]": l_cr,
            f"X-averaged {domain} particle crack length [m]": l_cr_av,
        }
        return variables

    def _get_mechanical_results(self, variables):
        domain_param = self.domain_param
        domain, Domain = self.domain_Domain

        c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
        c_s_surf = variables[f"{Domain} particle surface concentration [mol.m-3]"]
        T_xav = variables["X-averaged cell temperature [K]"]
        eps_s = variables[f"{Domain} electrode active material volume fraction"]

        if "Cell thickness change [m]" not in variables:
            # thermal expansion
            cell_thickness_change = T_xav * self.param.alpha_T_cell
        else:
            cell_thickness_change = variables["Cell thickness change [m]"]

        Omega = domain_param.Omega
        R0 = domain_param.prim.R
        c_0 = domain_param.c_0
        E0 = domain_param.E
        nu = domain_param.nu
        L0 = domain_param.L
        c_init = pybamm.r_average(domain_param.prim.c_init)
        v_change = pybamm.x_average(
            eps_s * domain_param.prim.t_change(c_s_rav)
        ) - pybamm.x_average(eps_s * domain_param.prim.t_change(c_init))

        cell_thickness_change += self.param.n_electrodes_parallel * v_change
        disp_surf_dim = Omega * R0 / 3 * (c_s_rav - c_0)
        # c0 reference concentration for no deformation
        # stress evaluated at the surface of the particles
        stress_r_surf = pybamm.Scalar(0)
        # c_s_rav is already multiplied by 3/R^3
        stress_t_surf = Omega * E0 / 3.0 / (1.0 - nu) * (c_s_rav - c_s_surf)

        return {
            f"{Domain} particle surface tangential stress [Pa]": stress_t_surf,
            f"{Domain} particle surface radial stress [Pa]": stress_r_surf,
            f"{Domain} particle surface displacement [m]": disp_surf,
            f"X-averaged {domain} particle surface "
            "radial stress [Pa]": stress_r_surf_av,
            f"X-averaged {domain} particle surface "
            "tangential stress [Pa]": stress_t_surf_av,
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
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        l_cr = variables[f"{Domain} particle crack length [m]"]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        R0 = self.domain_param.prim.R
        rho_cr = self.domain_param.rho_cr
        roughness = l_cr * 2 * rho_cr + 1  # the ratio of cracks to normal surface
        a_cr = (roughness - 1) * a  # normalised crack surface area
        a_cr_dim = a_cr / R0  # crack surface area to volume ratio [m-1]

        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            f"{Domain} crack surface to volume ratio [m-1]": a_cr_dim,
            f"{Domain} crack surface to volume ratio": a_cr,
            f"{Domain} electrode roughness ratio": roughness,
            f"X-averaged {domain} electrode roughness ratio": roughness_xavg,
        }
        return variables
