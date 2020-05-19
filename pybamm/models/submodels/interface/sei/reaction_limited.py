#
# Class for reaction limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class ReactionLimited(BaseModel):
    """Base class for reaction limited SEI growth.

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

        return variables

    def get_coupled_variables(self, variables):
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        try:
            j = variables[self.domain + " electrode interfacial current density"]
        except KeyError:
            j = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density"
            ]
        L_sei = variables["Total " + self.domain.lower() + " electrode sei thickness"]

        R_sei = pybamm.sei_parameters.R_sei
        alpha = 0.5
        # alpha = pybamm.sei_parameters.alpha
        if self.domain == "Negative":
            C_sei = pybamm.sei_parameters.C_sei_reaction_n

        # need to revise for thermal case
        j_sei = -(1 / C_sei) * pybamm.exp(
            -0.5 * (phi_s_n - phi_e_n - j * L_sei * R_sei)
        )

        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode sei interfacial current density" in variables
            and "Positive electrode sei interfacial current density" in variables
            and "Sei interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables["Inner " + domain + " sei thickness"]
        L_outer = variables["Outer " + domain + " sei thickness"]
        j_inner = variables["Inner " + domain + " sei interfacial current density"]
        j_outer = variables["Outer " + domain + " sei interfacial current density"]

        v_bar = pybamm.sei_parameters.v_bar
        if self.domain == "Negative":
            Gamma_SEI = pybamm.sei_parameters.Gamma_SEI_n

        self.rhs = {
            L_inner: -Gamma_SEI * j_inner,
            L_outer: -v_bar * Gamma_SEI * j_outer,
        }

    def set_initial_conditions(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables["Inner " + domain + " sei thickness"]
        L_outer = variables["Outer " + domain + " sei thickness"]

        L_inner_0 = pybamm.sei_parameters.L_inner_0
        L_outer_0 = pybamm.sei_parameters.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
