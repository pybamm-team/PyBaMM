#
# Base class for particle cracking models.
#
import pybamm


class BaseMechanics(pybamm.BaseSubModel):
    """
    Base class for particle mechanics models, referenced from [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : dict, optional
        Dictionary of either the electrode for "positive" or "Negative"
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
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)

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
        sto_rav = variables[f"R-averaged {domain} particle concentration"]
        c_s_surf = variables[f"{Domain} particle surface concentration [mol.m-3]"]
        T_xav = variables["X-averaged cell temperature [K]"]
        eps_s = variables[f"{Domain} electrode active material volume fraction"]

        Omega = domain_param.Omega
        R0 = domain_param.prim.R
        c_0 = domain_param.c_0
        E0 = domain_param.E
        nu = domain_param.nu
        sto_init = pybamm.r_average(domain_param.prim.c_init / domain_param.prim.c_max)
        v_change = pybamm.x_average(
            eps_s * domain_param.prim.t_change(sto_rav)
        ) - pybamm.x_average(eps_s * domain_param.prim.t_change(sto_init))

        electrode_thickness_change = self.param.n_electrodes_parallel * v_change
        # Ai2019 eq [10]
        disp_surf = Omega * R0 / 3 * (c_s_rav - c_0)
        # c0 reference concentration for no deformation
        # stress evaluated at the surface of the particles
        # Ai2019 eq [7] with r=R
        stress_r_surf = pybamm.Scalar(0)
        # Ai2019 eq [8] with r=R
        # c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = Omega * E0 / 3.0 / (1.0 - nu) * (c_s_rav - c_s_surf)

        # Averages
        stress_r_surf_av = pybamm.x_average(stress_r_surf)
        stress_t_surf_av = pybamm.x_average(stress_t_surf)
        disp_surf_av = pybamm.x_average(disp_surf)

        variables.update(
            {
                f"{Domain} particle surface radial stress [Pa]": stress_r_surf,
                f"{Domain} particle surface tangential stress [Pa]": stress_t_surf,
                f"{Domain} particle surface displacement [m]": disp_surf,
                f"X-averaged {domain} particle surface "
                "radial stress [Pa]": stress_r_surf_av,
                f"X-averaged {domain} particle surface "
                "tangential stress [Pa]": stress_t_surf_av,
                f"X-averaged {domain} particle surface displacement [m]": disp_surf_av,
                f"{Domain} electrode thickness change [m]": electrode_thickness_change,
            }
        )

        if (
            "Negative electrode thickness change [m]" in variables
            and "Positive electrode thickness change [m]" in variables
        ):
            # thermal expansion
            # Ai2019 eq [13]
            thermal_expansion = self.param.alpha_T_cell * (T_xav - self.param.T_ref)
            # calculate total cell thickness change
            neg_thickness_change = variables["Negative electrode thickness change [m]"]
            pos_thickness_change = variables["Positive electrode thickness change [m]"]
            variables["Cell thickness change [m]"] = (
                neg_thickness_change + pos_thickness_change + thermal_expansion
            )

        return variables

    def _get_standard_surface_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        l_cr = variables[f"{Domain} particle crack length [m]"]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        rho_cr = self.domain_param.rho_cr
        w_cr = self.domain_param.w_cr
        roughness = 1 + 2 * l_cr * rho_cr * w_cr  # ratio of cracks to normal surface
        a_cr = (roughness - 1) * a  # crack surface area to volume ratio

        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            f"{Domain} crack surface to volume ratio [m-1]": a_cr,
            f"{Domain} electrode roughness ratio": roughness,
            f"X-averaged {domain} electrode roughness ratio": roughness_xavg,
        }
        return variables
