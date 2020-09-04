#
# Class for reaction limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class EcReactionLimited(BaseModel):
    """
    Class for reaction limited SEI growth. This model assumes the "inner"
    SEI layer is of zero thickness and only models the "outer" SEI layer.

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

        L_inner = pybamm.FullBroadcast(
            0, self.domain.lower() + " electrode", "current collector"
        )
        L_outer = pybamm.standard_variables.L_outer

        j_inner = pybamm.FullBroadcast(
            0, self.domain.lower() + " electrode", "current collector"
        )
        j_outer = pybamm.Variable(
            "Outer " + self.domain + " electrode sei interfacial current density",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        variables = self._get_standard_thickness_variables(L_inner, L_outer)
        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        return variables

    def get_coupled_variables(self, variables):

        # Get variables related to the concentration
        variables.update(self._get_standard_concentraion_variables(variables))

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
        L_sei = variables["Outer " + domain + " sei thickness"]
        j_sei = variables["Outer " + domain + " sei interfacial current density"]

        if self.domain == "Negative":
            Gamma_SEI = self.param.Gamma_SEI_n

        self.rhs = {L_sei: -Gamma_SEI * j_sei / 2}

    def set_algebraic(self, variables):
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]
        j_sei = variables[
            "Outer "
            + self.domain.lower()
            + " electrode sei interfacial current density"
        ]
        L_sei = variables["Outer " + self.domain.lower() + " electrode sei thickness"]
        c_ec = variables[self.domain + " electrode EC surface concentration"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        try:
            j = variables[
                "Total "
                + self.domain.lower()
                + " electrode interfacial current density"
            ]
        except KeyError:
            j = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density"
            ]

        if self.domain == "Negative":
            C_sei_ec = self.param.C_sei_ec_n
            R_sei = self.param.R_sei_n

        # need to revise for thermal case
        self.algebraic = {
            j_sei: j_sei
            + C_sei_ec
            * c_ec
            * pybamm.exp(-0.5 * (phi_s_n - phi_e_n - j * L_sei * R_sei))
        }

    def set_initial_conditions(self, variables):
        L_sei = variables["Outer " + self.domain.lower() + " electrode sei thickness"]
        j_sei = variables[
            "Outer "
            + self.domain.lower()
            + " electrode sei interfacial current density"
        ]

        L_sei_0 = pybamm.Scalar(1)
        j_sei_0 = pybamm.Scalar(0)

        self.initial_conditions = {L_sei: L_sei_0, j_sei: j_sei_0}
