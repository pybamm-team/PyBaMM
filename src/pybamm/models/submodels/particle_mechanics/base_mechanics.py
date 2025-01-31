#
# Base class for particle cracking models.
#
import pybamm


class BaseMechanics(pybamm.BaseSubModel):
    """
    Base class for particle mechanics models, referenced from :footcite:t:`Ai2019` and
    :footcite:t:`Deshpande2012`.

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

    """

    def __init__(self, param, domain, options, phase="primary"):
        if options["particle size"] == "distribution":
            self.size_distribution = True
        else:
            self.size_distribution = False
        super().__init__(param, domain, options=options, phase=phase)

    def _get_standard_variables(self, l_cr):
        domain, Domain = self.domain_Domain
        l_cr_av = pybamm.x_average(l_cr)
        variables = {
            f"{Domain} {self.phase_param.phase_name}particle crack length [m]": l_cr,
            f"X-averaged {domain} {self.phase_param.phase_name}particle crack length [m]": l_cr_av,
        }
        return variables

    def _get_standard_size_distribution_variables(self, l_cr_dist):
        domain, Domain = self.domain_Domain
        l_cr_av_dist = pybamm.x_average(l_cr_dist)
        variables = {
            f"{Domain} {self.phase_param.phase_name}particle crack length distribution [m]": l_cr_dist,
            f"X-averaged {domain} {self.phase_param.phase_name}particle crack length distribution [m]": l_cr_av_dist,
        }
        return variables

    def _get_mechanical_size_distribution_results(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        phase_param = self.phase_param
        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration distribution [mol.m-3]"
        ]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration distribution [mol.m-3]"
        ]
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle size"],
        )
        T = pybamm.PrimaryBroadcast(
            T,
            [f"{domain} {phase_name}particle"],
        )

        # use a tangential approximation for omega
        c_0 = phase_param.c_0
        R0 = phase_param.R
        sto = variables[f"{Domain} {phase_name}particle concentration distribution"]
        Omega = pybamm.r_average(phase_param.Omega(sto, T))

        E0 = pybamm.r_average(phase_param.E(sto, T))
        nu = phase_param.nu
        # Ai2019 eq [10]
        disp_surf = Omega * R0 / 3 * (c_s_rav - c_0)
        # c0 reference concentration for no deformation
        # stress evaluated at the surface of the particles
        # Ai2019 eq [7] with r=R
        stress_r_surf = pybamm.Scalar(0)
        # Ai2019 eq [8] with r=R
        # c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = Omega * E0 * (c_s_rav - c_s_surf) / 3.0 / (1.0 - nu)

        # Averages
        stress_r_surf_av = pybamm.x_average(stress_r_surf)
        stress_t_surf_av = pybamm.x_average(stress_t_surf)
        disp_surf_av = pybamm.x_average(disp_surf)

        variables.update(
            {
                f"{Domain} {phase_name}particle surface radial stress distribution [Pa]": stress_r_surf,
                f"{Domain} {phase_name}particle surface tangential stress distribution [Pa]": stress_t_surf,
                f"{Domain} {phase_name}particle surface displacement distribution [m]": disp_surf,
                f"X-averaged {domain} {phase_name}particle surface "
                "radial stress distribution [Pa]": stress_r_surf_av,
                f"X-averaged {domain} {phase_name}particle surface "
                "tangential stress distribution [Pa]": stress_t_surf_av,
                f"X-averaged {domain} {phase_name}particle surface displacement distribution [m]": disp_surf_av,
            }
        )
        return variables

    def _get_mechanical_results(self, variables):
        domain_param = self.domain_param
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        sto_rav = variables[f"R-averaged {domain} {phase_name}particle concentration"]
        c_s_surf = variables[
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]"
        ]
        T_xav = variables["X-averaged cell temperature [K]"]
        T = pybamm.PrimaryBroadcast(
            variables[f"{Domain} electrode temperature [K]"],
            [f"{domain} {phase_name}particle"],
        )
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]

        # use a tangential approximation for omega
        c_0 = phase_param.c_0
        R0 = phase_param.R
        sto = variables[f"{Domain} {phase_name}particle concentration"]
        Omega = pybamm.r_average(phase_param.Omega(sto, T))

        E0 = pybamm.r_average(phase_param.E(sto, T))
        nu = phase_param.nu
        L0 = domain_param.L
        sto_init = pybamm.r_average(phase_param.c_init / phase_param.c_max)
        v_change = pybamm.x_average(
            eps_s * phase_param.t_change(sto_rav)
        ) - pybamm.x_average(eps_s * phase_param.t_change(sto_init))

        electrode_thickness_change = self.param.n_electrodes_parallel * v_change * L0
        # Ai2019 eq [10]
        disp_surf = Omega * R0 / 3 * (c_s_rav - c_0)
        # c0 reference concentration for no deformation
        # stress evaluated at the surface of the particles
        # Ai2019 eq [7] with r=R
        stress_r_surf = pybamm.Scalar(0)
        # Ai2019 eq [8] with r=R
        # c_s_rav is already multiplied by 3/R^3 inside r_average
        stress_t_surf = Omega * E0 * (c_s_rav - c_s_surf) / 3.0 / (1.0 - nu)

        # Averages
        stress_r_surf_av = pybamm.x_average(stress_r_surf)
        stress_t_surf_av = pybamm.x_average(stress_t_surf)
        disp_surf_av = pybamm.x_average(disp_surf)

        variables.update(
            {
                f"{Domain} {phase_name}particle surface radial stress [Pa]": stress_r_surf,
                f"{Domain} {phase_name}particle surface tangential stress [Pa]": stress_t_surf,
                f"{Domain} {phase_name}particle surface displacement [m]": disp_surf,
                f"X-averaged {domain} {phase_name}particle surface "
                "radial stress [Pa]": stress_r_surf_av,
                f"X-averaged {domain} {phase_name}particle surface "
                "tangential stress [Pa]": stress_t_surf_av,
                f"X-averaged {domain} {phase_name}particle surface displacement [m]": disp_surf_av,
                f"{Domain} electrode {phase_name}thickness change [m]": electrode_thickness_change,
            }
        )

        if (
            f"{Domain} primary thickness change [m]" in variables
            and f"{Domain} secondary thickness change [m]" in variables
        ):
            variables[f"{Domain} thickness change [m]"] = (
                variables[f"{Domain} primary thickness change [m]"]
                + variables[f"{Domain} secondary thickness change [m]"]
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
        phase_name = self.phase_param.phase_name

        l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
        a = variables[
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
        ]
        rho_cr = self.phase_param.rho_cr
        w_cr = self.phase_param.w_cr
        roughness = 1 + 2 * l_cr * rho_cr * w_cr  # ratio of cracks to normal surface
        a_cr = (roughness - 1) * a  # crack surface area to volume ratio

        roughness_xavg = pybamm.x_average(roughness)
        variables = {
            f"{Domain} {phase_name}crack surface to volume ratio [m-1]": a_cr,
            f"{Domain} {phase_name}electrode roughness ratio": roughness,
            f"X-averaged {domain} {phase_name}electrode roughness ratio": roughness_xavg,
        }
        return variables
