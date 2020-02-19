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


    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        L_inner = pybamm.standard_variables.L_inner
        L_outer = pybamm.standard_variables.L_outer

        variables = self._get_standard_thickness_variables(L_inner, L_outer)

        return variables

    def get_coupled_variables(self, variables):
        phi_s_n = variables["Negative electrode potential"]
        phi_e_n = variables["Negative electrolyte potential"]
        j_n = variables["Negative electrode interfacial current density"]
        L_sei = variables["Total SEI thickness"]

        C_sei = pybamm.sei_parameters.C_sei_reaction
        R_sei = pybamm.sei_parameters.R_sei
        alpha = 0.5
        # alpha = pybamm.sei_parameters.alpha

        # need to revise for thermal case
        j_sei = (1 / C_sei) * pybamm.exp(-(phi_s_n - phi_e_n - j_n * L_sei * R_sei))

        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        return variables

    def set_rhs(self, variables):
        L_inner = variables["Inner SEI thickness"]
        L_outer = variables["Outer SEI thickness"]
        j_inner = variables["Inner SEI reaction interfacial current density"]
        j_outer = variables["Outer SEI reaction interfacial current density"]

        v_bar = pybamm.sei_parameters.v_bar

        self.rhs = {L_inner: j_inner, L_outer: v_bar * j_outer}

    def set_initial_conditions(self, variables):
        L_inner = variables["Inner SEI thickness"]
        L_outer = variables["Outer SEI thickness"]

        L_inner_0 = pybamm.sei_parameters.L_inner_0
        L_outer_0 = pybamm.sei_parameters.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
