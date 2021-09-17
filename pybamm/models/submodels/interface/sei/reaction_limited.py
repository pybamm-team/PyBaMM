#
# Class for reaction limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class ReactionLimited(BaseModel):
    """
    Class for reaction limited SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, x_average):
        super().__init__(param)
        self.x_average = x_average

    def get_fundamental_variables(self):
        if self.x_average is True:
            L_inner_av = pybamm.standard_variables.L_inner_av
            L_outer_av = pybamm.standard_variables.L_outer_av
            L_inner = pybamm.PrimaryBroadcast(L_inner_av, "negative electrode")
            L_outer = pybamm.PrimaryBroadcast(L_outer_av, "negative electrode")
        else:
            L_inner = pybamm.standard_variables.L_inner
            L_outer = pybamm.standard_variables.L_outer

        variables = self._get_standard_thickness_variables(L_inner, L_outer)
        variables.update(self._get_standard_concentration_variables(variables))

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        phi_s_n = variables["Negative electrode potential"]
        phi_e_n = variables["Negative electrolyte potential"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        if "Negative electrode interfacial current density" in variables:
            j = variables["Negative electrode interfacial current density"]
        else:
            j = variables[
                "X-averaged negative electrode total interfacial current density"
            ]
        L_sei = variables["Total SEI thickness"]

        R_sei = self.param.R_sei
        alpha = 0.5
        # alpha = param.alpha
        C_sei = param.C_sei_reaction

        # need to revise for thermal case
        j_sei = -(1 / C_sei) * pybamm.exp(
            -0.5 * (phi_s_n - phi_e_n - j * L_sei * R_sei)
        )

        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        if self.x_average is True:
            L_inner = variables["X-averaged inner SEI thickness"]
            L_outer = variables["X-averaged outer SEI thickness"]
            j_inner = variables["X-averaged inner SEI interfacial current density"]
            j_outer = variables["X-averaged outer SEI interfacial current density"]
        else:
            L_inner = variables["Inner SEI thickness"]
            L_outer = variables["Outer SEI thickness"]
            j_inner = variables["Inner SEI interfacial current density"]
            j_outer = variables["Outer SEI interfacial current density"]

        v_bar = self.param.v_bar
        Gamma_SEI = self.param.Gamma_SEI

        self.rhs = {
            L_inner: -Gamma_SEI * j_inner,
            L_outer: -v_bar * Gamma_SEI * j_outer,
        }

    def set_initial_conditions(self, variables):
        if self.x_average is True:
            L_inner = variables["X-averaged inner SEI thickness"]
            L_outer = variables["X-averaged outer SEI thickness"]
        else:
            L_inner = variables["Inner SEI thickness"]
            L_outer = variables["Outer SEI thickness"]
        L_inner_0 = self.param.L_inner_0
        L_outer_0 = self.param.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
