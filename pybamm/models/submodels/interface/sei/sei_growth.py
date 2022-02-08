#
# Class for SEI growth
#
import pybamm
from .base_sei import BaseModel


class SEIGrowth(BaseModel):
    """
    Class for SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, reaction_loc, options=None):
        super().__init__(param, options=options)
        self.reaction_loc = reaction_loc

    def get_fundamental_variables(self):
        if self.reaction_loc == "x-average":
            L_inner_av = pybamm.standard_variables.L_inner_av
            L_outer_av = pybamm.standard_variables.L_outer_av
            L_inner = pybamm.PrimaryBroadcast(L_inner_av, "negative electrode")
            L_outer = pybamm.PrimaryBroadcast(L_outer_av, "negative electrode")
        elif self.reaction_loc == "full electrode":
            L_inner = pybamm.standard_variables.L_inner
            L_outer = pybamm.standard_variables.L_outer
        elif self.reaction_loc == "interface":
            L_inner = pybamm.standard_variables.L_inner_interface
            L_outer = pybamm.standard_variables.L_outer_interface

        if self.options["SEI"] == "ec reaction limited":
            L_inner = 0 * L_inner  # Set L_inner to zero, copying domains

        variables = self._get_standard_thickness_variables(L_inner, L_outer)
        variables.update(self._get_standard_concentration_variables(variables))

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        # delta_phi = phi_s - phi_e
        if self.reaction_loc == "interface":
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]
            phi_s_n = variables["Lithium metal interface electrode potential"]
        else:
            delta_phi = variables["Negative electrode surface potential difference"]
            phi_s_n = variables["Negative electrode potential"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        if "Negative electrode interfacial current density" in variables:
            j = variables["Negative electrode interfacial current density"]
        elif self.reaction_loc == "interface":
            j = variables["Lithium metal total interfacial current density"]
        else:
            j = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density"
            ]

        L_sei_inner = variables["Inner SEI thickness"]
        L_sei_outer = variables["Outer SEI thickness"]
        L_sei = variables["Total SEI thickness"]

        R_sei = self.param.R_sei

        if self.options["SEI"] == "reaction limited":
            # alpha = param.alpha
            C_sei = param.C_sei_reaction

            # need to revise for thermal case
            j_sei = -(1 / C_sei) * pybamm.exp(-0.5 * (delta_phi - j * L_sei * R_sei))

        elif self.options["SEI"] == "electron-migration limited":
            U_inner = self.param.U_inner_electron
            C_sei = self.param.C_sei_electron
            j_sei = (phi_s_n - U_inner) / (C_sei * L_sei_inner)

        elif self.options["SEI"] == "interstitial-diffusion limited":
            C_sei = self.param.C_sei_inter
            j_sei = -pybamm.exp(-delta_phi) / (C_sei * L_sei_inner)

        elif self.options["SEI"] == "solvent-diffusion limited":
            C_sei = self.param.C_sei_solvent
            j_sei = -1 / (C_sei * L_sei_outer)

        elif self.options["SEI"] == "ec reaction limited":
            C_sei_ec = self.param.C_sei_ec
            C_ec = self.param.C_ec

            # we have a linear system for j_sei and c_ec
            #  c_ec = 1 + j_sei * L_sei * C_ec
            #  j_sei = - C_sei_ec * c_ec * exp()
            # so
            #  j_sei = - C_sei_ec * exp() - j_sei * L_sei * C_ec * C_sei_ec * exp()
            # so
            #  j_sei = -C_sei_ec * exp() / (1 + L_sei * C_ec * C_sei_ec * exp())
            #  c_ec = 1 / (1 + L_sei * C_ec * C_sei_ec * exp())
            # need to revise for thermal case
            C_sei_exp = C_sei_ec * pybamm.exp(-0.5 * (delta_phi - j * L_sei * R_sei))
            j_sei = -C_sei_exp / (1 + L_sei * C_ec * C_sei_exp)
            c_ec = 1 / (1 + L_sei * C_ec * C_sei_exp)

            # Get variables related to the concentration
            c_ec_av = pybamm.x_average(c_ec)
            c_ec_scale = self.param.c_ec_0_dim

            variables.update(
                {
                    "EC surface concentration": c_ec,
                    "EC surface concentration [mol.m-3]": c_ec * c_ec_scale,
                    "X-averaged EC surface concentration": c_ec_av,
                    "X-averaged EC surface concentration [mol.m-3]": c_ec_av
                    * c_ec_scale,
                }
            )

        if self.options["SEI"] == "ec reaction limited":
            alpha = 0
        else:
            alpha = 0.5

        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        if self.reaction_loc == "x-average":
            L_inner = variables["X-averaged inner SEI thickness"]
            L_outer = variables["X-averaged outer SEI thickness"]
            j_inner = variables["X-averaged inner SEI interfacial current density"]
            j_outer = variables["X-averaged outer SEI interfacial current density"]
            # Note a is dimensionless (has a constant value of 1 if the surface
            # area does not change)
            a = variables["X-averaged negative electrode surface area to volume ratio"]
        else:
            L_inner = variables["Inner SEI thickness"]
            L_outer = variables["Outer SEI thickness"]
            j_inner = variables["Inner SEI interfacial current density"]
            j_outer = variables["Outer SEI interfacial current density"]
            if self.reaction_loc == "interface":
                a = 1
            else:
                a = variables["Negative electrode surface area to volume ratio"]

        Gamma_SEI = self.param.Gamma_SEI

        if self.options["SEI"] == "ec reaction limited":
            self.rhs = {L_outer: -Gamma_SEI * a * j_outer / 2}
        else:
            v_bar = self.param.v_bar
            self.rhs = {
                L_inner: -Gamma_SEI * a * j_inner,
                L_outer: -v_bar * Gamma_SEI * a * j_outer,
            }

    def set_initial_conditions(self, variables):
        if self.reaction_loc == "x-average":
            L_inner = variables["X-averaged inner SEI thickness"]
            L_outer = variables["X-averaged outer SEI thickness"]
        else:
            L_inner = variables["Inner SEI thickness"]
            L_outer = variables["Outer SEI thickness"]

        L_inner_0 = self.param.L_inner_0
        L_outer_0 = self.param.L_outer_0
        if self.options["SEI"] == "ec reaction limited":
            self.initial_conditions = {L_outer: L_inner_0 + L_outer_0}
        else:
            self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
