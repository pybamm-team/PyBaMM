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
        phase_name = self.phase_name
        phase_param = self.phase_param

        c_s_rav = pybamm.CoupledVariable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_s_rav.name: c_s_rav})

        c_s_surf = pybamm.CoupledVariable(
            f"{Domain} {phase_name}particle surface concentration [mol.m-3]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_s_surf.name: c_s_surf})

        T_xav = pybamm.CoupledVariable(
            "X-averaged cell temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T_xav.name: T_xav})

        T_electrode = pybamm.CoupledVariable(
            f"{Domain} electrode temperature [K]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({T_electrode.name: T_electrode})

        T = pybamm.PrimaryBroadcast(
            T_electrode,
            [f"{domain} {phase_name}particle"],
        )
        eps_s = pybamm.CoupledVariable(
            f"{Domain} electrode {phase_name}active material volume fraction",
            domain=f"{domain} electrode",
        )
        self.coupled_variables.update({eps_s.name: eps_s})

        # use a tangential approximation for omega
        sto = pybamm.CoupledVariable(
            f"{Domain} {phase_name}particle concentration",
            domains={
                "primary": f"{domain} particle",
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
        )
        self.coupled_variables.update({sto.name: sto})

        sto_rav = pybamm.CoupledVariable(
            f"R-averaged {domain} {phase_name}particle concentration",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({sto_rav.name: sto_rav})

        Omega = pybamm.r_average(phase_param.Omega(sto, T))
        R0 = phase_param.R
        c_0 = domain_param.c_0
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
        stress_t_surf = Omega * E0 / 3.0 / (1.0 - nu) * (c_s_rav - c_s_surf)

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

        # if (
        #    f"{Domain} primary thickness change [m]" in variables
        #    and f"{Domain} secondary thickness change [m]" in variables
        # ):
        if False:
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
        phase_name = self.phase_name

        l_cr = variables[f"{Domain} particle crack length [m]"]
        a = pybamm.CoupledVariable(
            f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({a.name: a})
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
