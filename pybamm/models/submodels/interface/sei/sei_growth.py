#
# Class for SEI growth
#
import pybamm
from .base_sei import BaseModel


class SEIGrowth(BaseModel):
    """
    Class for SEI growth.

    Most of the models are from section 5.6.4 of the thesis of
    Scott Marquis (Marquis, S. G. (2020). Long-term degradation of lithium-ion batteries
    (Doctoral dissertation, University of Oxford)), and references therein

    The ec reaction limited model is from: Yang, Xiao-Guang, et al. "Modeling of lithium
    plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
    aging." Journal of Power Sources 360 (2017): 28-40.

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

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, reaction_loc, options, phase="primary", cracks=False):
        super().__init__(param, options=options, phase=phase, cracks=cracks)
        self.reaction_loc = reaction_loc
        if self.options["SEI"] == "ec reaction limited":
            pybamm.citations.register("Yang2017")
        else:
            pybamm.citations.register("Marquis2020")

    def get_fundamental_variables(self):
        Ls = []
        for pos in ["inner", "outer"]:
            Pos = pos.capitalize()
            scale = self.phase_param.L_sei_0
            if self.reaction_loc == "x-average":
                L_av = pybamm.Variable(
                    f"X-averaged {pos} {self.reaction_name}thickness [m]",
                    domain="current collector",
                    scale=scale,
                )
                L_av.print_name = f"L_{pos}_av"
                L = pybamm.PrimaryBroadcast(L_av, "negative electrode")
            elif self.reaction_loc == "full electrode":
                L = pybamm.Variable(
                    f"{Pos} {self.reaction_name}thickness [m]",
                    domain="negative electrode",
                    auxiliary_domains={"secondary": "current collector"},
                    scale=scale,
                )
            elif self.reaction_loc == "interface":
                L = pybamm.Variable(
                    f"{Pos} {self.reaction_name}thickness [m]",
                    domain="current collector",
                    scale=scale,
                )
            L.print_name = f"L_{pos}"
            Ls.append(L)

        L_inner, L_outer = Ls

        if self.options["SEI"].startswith("ec reaction limited"):
            L_inner = 0 * L_inner  # Set L_inner to zero, copying domains

        variables = self._get_standard_thickness_variables(L_inner, L_outer)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        phase_param = self.phase_param
        # delta_phi = phi_s - phi_e
        T = variables["Negative electrode temperature [K]"]
        if self.reaction_loc == "interface":
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]
            T = pybamm.boundary_value(T, "right")
        else:
            delta_phi = variables["Negative electrode surface potential difference [V]"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        if "Negative electrode interfacial current density [A.m-2]" in variables:
            j = variables["Negative electrode interfacial current density [A.m-2]"]
        elif self.reaction_loc == "interface":
            j = variables["Lithium metal total interfacial current density [A.m-2]"]
        else:
            j = variables[
                "X-averaged negative electrode total "
                "interfacial current density [A.m-2]"
            ]

        L_sei_inner = variables[f"Inner {self.reaction_name}thickness [m]"]
        L_sei_outer = variables[f"Outer {self.reaction_name}thickness [m]"]
        L_sei = variables[f"Total {self.reaction_name}thickness [m]"]

        R_sei = phase_param.R_sei
        eta_SEI = delta_phi - phase_param.U_sei - j * L_sei * R_sei
        # Thermal prefactor for reaction, interstitial and EC models
        F_RT = param.F / (param.R * T)

        # Define alpha_SEI depending on whether it is symmetric or asymmetric. This
        # applies to "reaction limited" and "EC reaction limited"
        if self.options["SEI"].endswith("(asymmetric)"):
            alpha_SEI = phase_param.alpha_SEI
        else:
            alpha_SEI = 0.5

        if self.options["SEI"].startswith("reaction limited"):
            # Scott Marquis thesis (eq. 5.92)
            j_sei = -phase_param.j0_sei * pybamm.exp(-alpha_SEI * F_RT * eta_SEI)

        elif self.options["SEI"] == "electron-migration limited":
            # Scott Marquis thesis (eq. 5.94)
            eta_inner = delta_phi - phase_param.U_inner
            j_sei = phase_param.kappa_inner * eta_inner / L_sei_inner

        elif self.options["SEI"] == "interstitial-diffusion limited":
            # Scott Marquis thesis (eq. 5.96)
            j_sei = -(
                phase_param.D_li * phase_param.c_li_0 * param.F / L_sei_outer
            ) * pybamm.exp(-F_RT * delta_phi)

        elif self.options["SEI"] == "solvent-diffusion limited":
            # Scott Marquis thesis (eq. 5.91)
            j_sei = -phase_param.D_sol * phase_param.c_sol * param.F / L_sei_outer

        elif self.options["SEI"].startswith("ec reaction limited"):
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
                name = "EC concentration on cracks [mol.m-3]"
            else:
                name = "EC surface concentration [mol.m-3]"
            variables.update({name: c_ec, f"X-averaged {name}": c_ec_av})

        if self.options["SEI"].startswith("ec reaction limited"):
            inner_sei_proportion = 0
        else:
            inner_sei_proportion = phase_param.inner_sei_proportion

        # All SEI growth mechanisms assumed to have Arrhenius dependence
        Arrhenius = pybamm.exp(phase_param.E_sei / param.R * (1 / param.T_ref - 1 / T))

        j_inner = inner_sei_proportion * Arrhenius * j_sei
        j_outer = (1 - inner_sei_proportion) * Arrhenius * j_sei

        variables.update(self._get_standard_concentration_variables(variables))
        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Add other standard coupled variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        phase_param = self.phase_param
        param = self.param

        if self.reaction_loc == "x-average":
            L_inner = variables[f"X-averaged inner {self.reaction_name}thickness [m]"]
            L_outer = variables[f"X-averaged outer {self.reaction_name}thickness [m]"]
            j_inner = variables[
                f"X-averaged inner {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]
            j_outer = variables[
                f"X-averaged outer {self.reaction_name}"
                "interfacial current density [A.m-2]"
            ]

        else:
            L_inner = variables[f"Inner {self.reaction_name}thickness [m]"]
            L_outer = variables[f"Outer {self.reaction_name}thickness [m]"]
            j_inner = variables[
                f"Inner {self.reaction_name}interfacial current density [A.m-2]"
            ]
            j_outer = variables[
                f"Outer {self.reaction_name}interfacial current density [A.m-2]"
            ]

        # The spreading term acts to spread out SEI along the cracks as they grow.
        # For SEI on initial surface (as opposed to cracks), it is zero.
        if self.reaction == "SEI on cracks":
            if self.reaction_loc == "x-average":
                l_cr = variables["X-averaged negative particle crack length [m]"]
                dl_cr = variables["X-averaged negative particle cracking rate [m.s-1]"]
            else:
                l_cr = variables["Negative particle crack length [m]"]
                dl_cr = variables["Negative particle cracking rate [m.s-1]"]
            spreading_outer = (
                dl_cr / l_cr * (self.phase_param.L_outer_crack_0 - L_outer)
            )
            spreading_inner = (
                dl_cr / l_cr * (self.phase_param.L_inner_crack_0 - L_inner)
            )
        else:
            spreading_outer = 0
            spreading_inner = 0

        # a * j_sei / F is the rate of consumption of li moles by SEI reaction
        # 1/z_sei converts from li moles to SEI moles (z_sei=li mol per sei mol)
        # a * j_sei / (F * z_sei) is the rate of consumption of SEI moles by SEI
        # reaction
        # V_bar / a converts from SEI moles to SEI thickness
        # V_bar * j_sei / (F * z_sei) is the rate of SEI thickness change
        dLdt_SEI_inner = (
            phase_param.V_bar_inner * j_inner / (param.F * phase_param.z_sei)
        )
        dLdt_SEI_outer = (
            phase_param.V_bar_outer * j_outer / (param.F * phase_param.z_sei)
        )

        # we have to add the spreading rate to account for cracking
        if self.options["SEI"].startswith("ec reaction limited"):
            self.rhs = {L_outer: -dLdt_SEI_outer + spreading_outer}
        else:
            self.rhs = {
                L_inner: -dLdt_SEI_inner + spreading_inner,
                L_outer: -dLdt_SEI_outer + spreading_outer,
            }

    def set_initial_conditions(self, variables):
        if self.reaction_loc == "x-average":
            L_inner = variables[f"X-averaged inner {self.reaction_name}thickness [m]"]
            L_outer = variables[f"X-averaged outer {self.reaction_name}thickness [m]"]
        else:
            L_inner = variables[f"Inner {self.reaction_name}thickness [m]"]
            L_outer = variables[f"Outer {self.reaction_name}thickness [m]"]

        if self.reaction == "SEI on cracks":
            L_inner_0 = self.phase_param.L_inner_crack_0
            L_outer_0 = self.phase_param.L_outer_crack_0
        else:
            L_inner_0 = self.phase_param.L_inner_0
            L_outer_0 = self.phase_param.L_outer_0
        if self.options["SEI"].startswith("ec reaction limited"):
            self.initial_conditions = {L_outer: L_inner_0 + L_outer_0}
        else:
            self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
