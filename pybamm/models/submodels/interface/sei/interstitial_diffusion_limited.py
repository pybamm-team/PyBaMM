#
# Class for interstitial-diffusion limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class InterstitialDiffusionLimited(BaseModel):
    """
    Class for interstitial-diffusion limited SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.sei.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        L_inner = pybamm.standard_variables.L_inner
        L_outer = pybamm.standard_variables.L_outer

        variables = self._get_standard_thickness_variables(L_inner, L_outer)
        variables.update(self._get_standard_concentration_variables(variables))

        return variables

    def get_coupled_variables(self, variables):
        L_sei_inner = variables[
            "Inner " + self.domain.lower() + " electrode SEI thickness"
        ]
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]

        if self.domain == "Negative":
            C_sei = self.param.C_sei_inter_n

            j_sei = -pybamm.exp(-(phi_s_n - phi_e_n)) / (C_sei * L_sei_inner)

        alpha = 0.5
        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode SEI interfacial current density" in variables
            and "Positive electrode SEI interfacial current density" in variables
            and "SEI interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables["Inner " + domain + " SEI thickness"]
        L_outer = variables["Outer " + domain + " SEI thickness"]
        j_inner = variables["Inner " + domain + " SEI interfacial current density"]
        j_outer = variables["Outer " + domain + " SEI interfacial current density"]

        v_bar = self.param.v_bar

        if self.domain == "Negative":
            Gamma_SEI = self.param.Gamma_SEI_n

        self.rhs = {
            L_inner: -Gamma_SEI * j_inner,
            L_outer: -v_bar * Gamma_SEI * j_outer,
        }

    def set_initial_conditions(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables["Inner " + domain + " SEI thickness"]
        L_outer = variables["Outer " + domain + " SEI thickness"]

        L_inner_0 = self.param.L_inner_0
        L_outer_0 = self.param.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
