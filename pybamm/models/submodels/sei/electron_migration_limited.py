#
# Class for electron-migration limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class ElectronMigrationLimited(BaseModel):
    """Base class for electron-migration limited SEI growth.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        L_inner = pybamm.standard_variables.L_inner
        L_outer = pybamm.standard_variables.L_outer

        variables = self._get_standard_thickness_variables(L_inner, L_outer)

        return variables

    def get_coupled_variables(self, variables):
        L_sei_inner = variables["Inner negative electrode sei thickness"]
        phi_s_n = variables["Negative electrode potential"]

        C_sei = pybamm.sei_parameters.C_sei_electron
        U_inner = pybamm.sei_parameters.U_inner_electron

        j_sei = (phi_s_n - U_inner) / (C_sei * L_sei_inner)

        alpha = 0.5
        j_inner = alpha * j_sei
        j_outer = (1 - alpha) * j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        return variables

    def set_rhs(self, variables):
        L_inner = variables["Inner " + self.domain.lower() + " sei thickness"]
        L_outer = variables["Outer " + self.domain.lower() + " sei thickness"]
        j_inner = variables[
            "Inner " + self.domain.lower() + " sei interfacial current density"
        ]
        j_outer = variables[
            "Outer " + self.domain.lower() + " sei interfacial current density"
        ]

        v_bar = pybamm.sei_parameters.v_bar

        self.rhs = {L_inner: -j_inner, L_outer: -v_bar * j_outer}

    def set_initial_conditions(self, variables):
        L_inner = variables["Inner " + self.domain.lower() + " sei thickness"]
        L_outer = variables["Outer " + self.domain.lower() + " sei thickness"]

        L_inner_0 = pybamm.sei_parameters.L_inner_0
        L_outer_0 = pybamm.sei_parameters.L_outer_0

        self.initial_conditions = {L_inner: L_inner_0, L_outer: L_outer_0}
