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
        else:
            pybamm.citations.register("Marquis2020")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        cs = []
        for pos in ["inner", "outer"]:
            if self.reaction_loc == "x-average":
                c_sei_0 = self.phase_param.a_typ * (
                    self.phase_param.L_inner_0 / self.phase_param.V_bar_inner
                    + self.phase_param.L_outer_0 / self.phase_param.V_bar_outer
                )
                c_av = pybamm.Variable(
                    f"X-averaged {domain} {pos} {self.reaction_name}"
                    "concentration [mol.m-3]",
                    domain="current collector",
                    scale=c_sei_0,
                )
                c_av.print_name = f"c_{pos}_av"
                c = pybamm.PrimaryBroadcast(c_av, "negative electrode")
            elif self.reaction_loc == "full electrode":
                c_sei_0 = self.phase_param.a_typ * (
                    self.phase_param.L_inner_0 / self.phase_param.V_bar_inner
                    + self.phase_param.L_outer_0 / self.phase_param.V_bar_outer
                )
                c = pybamm.Variable(
                    f"{Domain} {pos} {self.reaction_name}concentration [mol.m-3]",
                    domain="negative electrode",
                    auxiliary_domains={"secondary": "current collector"},
                    scale=c_sei_0,
                )
            elif self.reaction_loc == "interface":
                c_sei_0 = (
                    self.phase_param.L_inner_0 / self.phase_param.V_bar_inner
                    + self.phase_param.L_outer_0 / self.phase_param.V_bar_outer
                )
                c = pybamm.Variable(
                    f"{Domain} {pos} {self.reaction_name}concentration [mol.m-2]",
                    domain="current collector",
                    scale=c_sei_0,
                )
            c.print_name = f"c_{pos}"
            cs.append(c)

        c_inner, c_outer = cs

        SEI_option = getattr(self.options, domain)["SEI"]
        if SEI_option.startswith("ec reaction limited"):
            c_inner = 0 * c_inner  # Set L_inner to zero, copying domains

        variables = self._get_standard_concentration_variables(c_inner, c_outer)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        phase_param = self.phase_param
        domain, Domain = self.domain_Domain
        SEI_option = getattr(self.options, domain)["SEI"]
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

        L_sei_inner = variables[f"{Domain} inner {self.reaction_name}thickness [m]"]
        L_sei_outer = variables[f"{Domain} outer {self.reaction_name}thickness [m]"]
        L_sei = variables[f"{Domain} total {self.reaction_name}thickness [m]"]

        R_sei = phase_param.R_sei
        eta_SEI = delta_phi - phase_param.U_sei - j * L_sei * R_sei
        # Thermal prefactor for reaction, interstitial and EC models
        F_RT = param.F / (param.R * T)

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
            eta_inner = delta_phi - phase_param.U_inner
            j_sei = phase_param.kappa_inner * eta_inner / L_sei_inner

        elif SEI_option == "interstitial-diffusion limited":
            # Scott Marquis thesis (eq. 5.96)
            j_sei = -(
                phase_param.D_li * phase_param.c_li_0 * param.F / L_sei_outer
            ) * pybamm.exp(-F_RT * delta_phi)

        elif SEI_option == "solvent-diffusion limited":
            # Scott Marquis thesis (eq. 5.91)
            j_sei = -phase_param.D_sol * phase_param.c_sol * param.F / L_sei_outer

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
            j_sei = -param.F * c_0 * k_exp / (1 + L_over_D * k_exp)
            c_ec = c_0 / (1 + L_over_D * k_exp)

            # Get variables related to the concentration
            c_ec_av = pybamm.x_average(c_ec)

            if self.reaction == "SEI on cracks":
                name = f"{Domain} EC concentration on cracks [mol.m-3]"
            else:
                name = f"{Domain} EC surface concentration [mol.m-3]"
            variables.update({name: c_ec, f"X-averaged {name}": c_ec_av})

        if SEI_option.startswith("ec reaction limited"):
            inner_sei_proportion = 0
        else:
            inner_sei_proportion = phase_param.inner_sei_proportion

        # All SEI growth mechanisms assumed to have Arrhenius dependence
        Arrhenius = pybamm.exp(phase_param.E_sei / param.R * (1 / param.T_ref - 1 / T))

        j_inner = inner_sei_proportion * Arrhenius * j_sei
        j_outer = (1 - inner_sei_proportion) * Arrhenius * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Add other standard coupled variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        phase_param = self.phase_param
        param = self.param
        domain, Domain = self.domain_Domain

        if self.reaction_loc == "interface":
            c_inner = variables[
                f"{Domain} inner {self.reaction_name}concentration [mol.m-2]"
            ]
            c_outer = variables[
                f"{Domain} outer {self.reaction_name}concentration [mol.m-2]"
            ]
            j_inner = variables[
                f"{Domain} inner {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            j_outer = variables[
                f"{Domain} outer {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            a = 1
        elif self.reaction_loc == "x-average":
            c_inner = variables[
                f"X-averaged {domain} inner {self.reaction_name}concentration [mol.m-3]"
            ]
            c_outer = variables[
                f"X-averaged {domain} outer {self.reaction_name}concentration [mol.m-3]"
            ]
            j_inner = variables[
                f"X-averaged {domain} electrode inner {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            j_outer = variables[
                f"X-averaged {domain} electrode outer {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            a = variables[
                f"X-averaged {domain} electrode {self.phase_name}"
                "surface area to volume ratio [m-1]"
            ]
        else:
            c_inner = variables[
                f"{Domain} inner {self.reaction_name}concentration [mol.m-3]"
            ]
            c_outer = variables[
                f"{Domain} outer {self.reaction_name}concentration [mol.m-3]"
            ]
            j_inner = variables[
                f"{Domain} electrode inner {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            j_outer = variables[
                f"{Domain} electrode outer {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            a = variables[
                f"{Domain} electrode {self.phase_name}"
                "surface area to volume ratio [m-1]"
            ]

        if self.reaction == "SEI on cracks":
            if self.reaction_loc == "x-average":
                roughness = variables[f"X-averaged {domain} electrode roughness ratio"]
            else:
                roughness = variables[f"{Domain} electrode roughness ratio"]
            a *= roughness - 1  # Replace surface area with crack area

        # a * j_sei / F is the rate of consumption of li moles by SEI reaction
        # 1/z_sei converts from li moles to SEI moles (z_sei=li mol per sei mol)
        # a * j_sei / (F * z_sei) is the rate of consumption of SEI moles by SEI
        # reaction
        dcdt_SEI_inner = a * j_inner / (param.F * phase_param.z_sei)
        dcdt_SEI_outer = a * j_outer / (param.F * phase_param.z_sei)

        if self.options["SEI"].startswith("ec reaction limited"):
            self.rhs = {c_outer: -dcdt_SEI_outer}
        else:
            self.rhs = {c_inner: -dcdt_SEI_inner, c_outer: -dcdt_SEI_outer}

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        if self.reaction_loc == "interface":
            c_inner = variables[
                f"{Domain} inner {self.reaction_name}concentration [mol.m-2]"
            ]
            c_outer = variables[
                f"{Domain} outer {self.reaction_name}concentration [mol.m-2]"
            ]
            a = 1
        elif self.reaction_loc == "x-average":
            c_inner = variables[
                f"X-averaged {domain} inner {self.reaction_name}concentration [mol.m-3]"
            ]
            c_outer = variables[
                f"X-averaged {domain} outer {self.reaction_name}concentration [mol.m-3]"
            ]
            a = self.phase_param.a_typ
        else:
            c_inner = variables[
                f"{Domain} inner {self.reaction_name}concentration [mol.m-3]"
            ]
            c_outer = variables[
                f"{Domain} outer {self.reaction_name}concentration [mol.m-3]"
            ]
            a = self.phase_param.a_typ

        if self.reaction == "SEI on cracks":
            L_inner_0 = self.phase_param.L_inner_crack_0
            L_outer_0 = self.phase_param.L_outer_crack_0
            roughness_init = 1 + 2 * (
                self.param.n.l_cr_0 * self.param.n.rho_cr * self.param.n.w_cr
            )
            a *= roughness_init - 1  # Replace surface area with crack area
        else:
            L_inner_0 = self.phase_param.L_inner_0
            L_outer_0 = self.phase_param.L_outer_0

        c_inner_0 = a * L_inner_0 / self.phase_param.V_bar_inner
        c_outer_0 = a * L_outer_0 / self.phase_param.V_bar_outer
        SEI_option = getattr(self.options, domain)["SEI"]
        if SEI_option.startswith("ec reaction limited"):
            self.initial_conditions = {c_outer: c_inner_0 + c_outer_0}
        else:
            self.initial_conditions = {c_inner: c_inner_0, c_outer: c_outer_0}
