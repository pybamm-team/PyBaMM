#
# Class for SEI growth
#
import pybamm

from .base_sei import BaseModel


class SEIGrowth(BaseModel):
    """
    Class for SEI growth.

    Most of the models are from Section 5.6.4 of :footcite:t:`Marquis2020` and
    references therein.

    The ec reaction limited model is from :footcite:t:`Yang2017`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)
    options : dict
        A dictionary of options to be passed to the model.
    phase : str, optional
        Phase of the particle (default is "primary")
    cracks : bool, optional
        Whether this is a submodel for standard SEI or SEI on cracks
    """

    def __init__(
        self, param, domain, reaction_loc, options, phase="primary", cracks=False
    ):
        super().__init__(param, domain, options=options, phase=phase, cracks=cracks)
        self.reaction_loc = reaction_loc
        SEI_option = getattr(self.options, domain)["SEI"]
        if SEI_option == "ec reaction limited":
            pybamm.citations.register("Yang2017")
        elif SEI_option == "VonKolzenberg2020":
            pybamm.citations.register("VonKolzenberg2020")
        elif SEI_option == "tunnelling limited":
            pybamm.citations.register("Tang2012")
        else:
            pybamm.citations.register("Marquis2020")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        scale = self.phase_param.L_sei_0
        if self.reaction_loc == "x-average":
            L_sei_av = pybamm.Variable(
                f"X-averaged {domain} {self.reaction_name}thickness [m]",
                domain="current collector",
                scale=scale,
            )
            L_sei_av.print_name = "L_sei_av"
            L_sei = pybamm.PrimaryBroadcast(L_sei_av, f"{domain} electrode")
        elif self.reaction_loc == "full electrode":
            L_sei = pybamm.Variable(
                f"{Domain} {self.reaction_name}thickness [m]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=scale,
            )
        elif self.reaction_loc == "interface":
            L_sei = pybamm.Variable(
                f"{Domain} {self.reaction_name}thickness [m]",
                domain="current collector",
                scale=scale,
            )
        L_sei.print_name = "L_sei"

        variables = self._get_standard_thickness_variables(L_sei)

        return variables

    def get_coupled_variables(self, variables):
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain
        SEI_option = getattr(getattr(self.options, domain), self.phase)["SEI"]
        T = variables[f"{Domain} electrode temperature [K]"]
        # delta_phi = phi_s - phi_e
        if self.reaction_loc == "interface":
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]
            T = pybamm.boundary_value(T, "right")
        else:
            delta_phi = variables[
                f"{Domain} electrode surface potential difference [V]"
            ]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        if f"{Domain} electrode interfacial current density [A.m-2]" in variables:
            j = variables[f"{Domain} electrode interfacial current density [A.m-2]"]
        elif self.reaction_loc == "interface":
            j = variables["Lithium metal total interfacial current density [A.m-2]"]
        else:
            j = variables[
                f"X-averaged {domain} electrode total "
                "interfacial current density [A.m-2]"
            ]

        L_sei = variables[f"{Domain} total {self.reaction_name}thickness [m]"]

        R_sei = phase_param.R_sei
        eta_SEI = delta_phi - phase_param.U_sei - j * L_sei * R_sei
        # Thermal prefactor for reaction, interstitial and EC models
        F_RT = self.param.F / (self.param.R * T)

        # Define alpha_SEI depending on whether it is symmetric or asymmetric. This
        # applies to "reaction limited" and "EC reaction limited"
        if SEI_option.endswith("(asymmetric)"):
            alpha_SEI = phase_param.alpha_SEI
        else:
            alpha_SEI = 0.5

        if SEI_option.startswith("reaction limited"):
            # Scott Marquis thesis (eq. 5.92)
            j_sei = -phase_param.j0_sei * pybamm.exp(-alpha_SEI * F_RT * eta_SEI)

        elif SEI_option == "electron-migration limited":
            # Scott Marquis thesis (eq. 5.94)
            eta_inner = delta_phi - phase_param.U_sei
            j_sei = (eta_inner < 0) * phase_param.kappa_inner * eta_inner / L_sei

        elif SEI_option == "tunnelling limited":  #
            # This comes from eq.25 in Tang, M., Lu, S. and Newman, J., 2012.
            # Experimental and theoretical investigation of solid-electrolyte-interphase formation mechanisms on glassy carbon.
            # Journal of The Electrochemical Society, 159(11), p.A1775.
            j_sei = (
                -phase_param.j0_sei
                * pybamm.exp(-alpha_SEI * F_RT * delta_phi)
                * pybamm.exp(-phase_param.beta_tunnelling * L_sei)
            )

        elif SEI_option == "VonKolzenberg2020":
            # Equation 19 in
            # von Kolzenberg L, Latz A, Horstmann B.
            # Solid electrolyte interphase during battery cycling:
            # Theory of growth regimes. ChemSusChem. 2020 Aug 7;13(15):3901-10.
            eta_bar = F_RT * (delta_phi)
            L_diff = (
                phase_param.D_li
                * phase_param.c_li_0
                * self.param.F
                / phase_param.j0_sei
            ) * pybamm.exp(-(1 - alpha_SEI) * eta_bar)
            L_tun = phase_param.L_tunneling
            L_app = (L_sei - L_tun) * ((L_sei - L_tun) > 0)
            L_mig = (
                2
                / F_RT
                * phase_param.kappa_Li_ion
                / pybamm.maximum(pybamm.AbsoluteValue(j), 1e-38)
            )
            sign_j = 2 * (j > 0) - 1
            LL_k = (1 - L_app / L_mig * sign_j) / (
                1 - L_app / L_mig * sign_j + L_app / L_diff
            )
            j_sei = -phase_param.j0_sei * LL_k * pybamm.exp(-alpha_SEI * eta_bar)

        elif SEI_option == "interstitial-diffusion limited":
            # Scott Marquis thesis (eq. 5.96)
            j_sei = -(
                phase_param.D_li * phase_param.c_li_0 * self.param.F / L_sei
            ) * pybamm.exp(-F_RT * delta_phi)

        elif SEI_option == "solvent-diffusion limited":
            # Scott Marquis thesis (eq. 5.91)
            j_sei = -phase_param.D_sol * phase_param.c_sol * self.param.F / L_sei

        elif SEI_option.startswith("ec reaction limited"):
            # we have a linear system for j and c
            #  c = c_0 + j * L / F / D          [1] (eq 11 in the Yang2017 paper)
            #  j = - F * c * k_exp()            [2] (eq 10 in the Yang2017 paper, factor
            #                                        of a is outside the defn of j here)
            # [1] into [2] gives (F cancels in the second terms)
            #  j = - F * c_0 * k_exp() - j * L * k_exp() / D
            # rearrange
            #  j = -F * c_0* k_exp() / (1 + L * k_exp() / D)
            #  c_ec = c_0 - L * k_exp() / D / (1 + L * k_exp() / D)
            #       = c_0 / (1 + L * k_exp() / D)
            k_exp = phase_param.k_sei * pybamm.exp(-alpha_SEI * F_RT * eta_SEI)
            L_over_D = L_sei / phase_param.D_ec
            c_0 = phase_param.c_ec_0
            j_sei = -self.param.F * c_0 * k_exp / (1 + L_over_D * k_exp)
            c_ec = c_0 / (1 + L_over_D * k_exp)

            # Get variables related to the concentration
            c_ec_av = pybamm.x_average(c_ec)

            if self.reaction == "SEI on cracks":
                name = f"{Domain} EC concentration on cracks [mol.m-3]"
            else:
                name = f"{Domain} EC surface concentration [mol.m-3]"
            variables.update({name: c_ec, f"X-averaged {name}": c_ec_av})

        # All SEI growth mechanisms assumed to have Arrhenius dependence
        arrhenius = pybamm.exp(
            phase_param.E_sei / self.param.R * (1 / self.param.T_ref - 1 / T)
        )

        j_sei = arrhenius * j_sei

        variables.update(self._get_standard_concentration_variables(variables))
        variables.update(self._get_standard_reaction_variables(j_sei))

        # Add other standard coupled variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain

        if self.reaction_loc == "x-average":
            L_sei = variables[f"X-averaged {domain} {self.reaction_name}thickness [m]"]
            j_sei = variables[
                f"X-averaged {domain} electrode {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]

        else:
            L_sei = variables[f"{Domain} {self.reaction_name}thickness [m]"]
            j_sei = variables[
                f"{Domain} electrode {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]

        # The spreading term acts to spread out SEI along the cracks as they grow.
        # For SEI on initial surface (as opposed to cracks), it is zero.
        if self.reaction == "SEI on cracks":
            if self.reaction_loc == "x-average":
                l_cr = variables[f"X-averaged {domain} particle crack length [m]"]
                dl_cr = variables[f"X-averaged {domain} particle cracking rate [m.s-1]"]
            else:
                l_cr = variables[f"{Domain} particle crack length [m]"]
                dl_cr = variables[f"{Domain} particle cracking rate [m.s-1]"]
            spreading = dl_cr / l_cr * (self.phase_param.L_sei_crack_0 - L_sei)
        else:
            spreading = 0

        # a * j_sei / F is the rate of consumption of li moles by SEI reaction
        # 1/z_sei converts from li moles to SEI moles (z_sei=li mol per sei mol)
        # a * j_sei / (F * z_sei) is the rate of consumption of SEI moles by SEI
        # reaction
        # V_bar / a converts from SEI moles to SEI thickness
        # V_bar * j_sei / (F * z_sei) is the rate of SEI thickness change
        dLdt_SEI = phase_param.V_bar_sei * j_sei / (self.param.F * phase_param.z_sei)

        # we have to add the spreading rate to account for cracking
        self.rhs = {L_sei: -dLdt_SEI + spreading}

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        if self.reaction_loc == "x-average":
            L_sei = variables[f"X-averaged {domain} {self.reaction_name}thickness [m]"]
        else:
            L_sei = variables[f"{Domain} {self.reaction_name}thickness [m]"]

        if self.reaction == "SEI on cracks":
            L_sei_0 = self.phase_param.L_sei_crack_0
        else:
            L_sei_0 = self.phase_param.L_sei_0
        self.initial_conditions = {L_sei: L_sei_0}
