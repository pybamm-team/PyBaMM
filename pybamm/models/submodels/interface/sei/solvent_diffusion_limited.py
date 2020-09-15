#
# Class for solvent-diffusion limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class SolventDiffusionLimited(BaseModel):
    """
    Class for solvent-diffusion limited SEI growth.

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
        variables.update(self._get_standard_concentraion_variables(variables))

        return variables

    def get_coupled_variables(self, variables):
        L_sei_outer = variables[
            f"Outer {self.domain.lower()} electrode{self.reaction_name} thickness"
        ]

        if self.domain == "Negative":
            C_sei = self.param.C_sei_solvent_n

        j_sei = -1 / (C_sei * L_sei_outer)
        alpha = pybamm.Parameter("Inner SEI reaction proportion") #0.5
        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Update whole cell variables, which also updates the "sum of" variables
        if (
            f"Negative electrode{self.reaction_name} interfacial current density"
            in variables
            and f"Positive electrode{self.reaction_name} interfacial current density"
            in variables
            and f"{self.reaction.capitalize()} interfacial current density"
            not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables[f"Inner {domain}{self.reaction_name} thickness"]
        L_outer = variables[f"Outer {domain}{self.reaction_name} thickness"]
        j_inner = variables[
            f"Inner {domain}{self.reaction_name} interfacial current density"
        ]
        j_outer = variables[
            f"Outer {domain}{self.reaction_name} interfacial current density"
        ]
        # ratio of average sei thickness between before and after crack propagation
        if self.reaction_name == "sei-cracks":
            l_cr_n = variables[f"{self.domain} particle crack length"]
            dl_cr_n = variables[f"{self.domain} particle cracking rate"]
            ratio_avg =  l_cr_n / (l_cr_n + dl_cr_n)  
        else:
            ratio_avg = 1
        v_bar = self.param.v_bar

        if self.domain == "Negative":
            Gamma_SEI = self.param.Gamma_SEI_n

        self.rhs = {
            L_inner: -Gamma_SEI * j_inner / ratio_avg - L_inner * (1 - ratio_avg),
            L_outer: -v_bar * Gamma_SEI * j_outer / ratio_avg - L_outer * (1 - ratio_avg),
        }

    def set_initial_conditions(self, variables):
        domain = self.domain.lower() + " electrode"
        L_inner = variables[f"Inner {domain}{self.reaction_name} thickness"]
        L_outer = variables[f"Outer {domain}{self.reaction_name} thickness"]

        L_inner_0 = self.param.L_inner_0
        L_outer_0 = self.param.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
