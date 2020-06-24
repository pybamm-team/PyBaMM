#
# Class for reaction limited SEI growth
#
import pybamm
from .base_sei import BaseModel


class EcReactionLimited(BaseModel):
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

        L_sei = pybamm.Variable(
            "Total " + self.domain.lower() + " electrode sei thickness",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        j_sei = pybamm.Variable(
            self.domain + " electrode sei interfacial current density",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        variables = self._get_standard_total_thickness_variables(L_sei)
        variables.update(self._get_standard_total_reaction_variables(j_sei))

        return variables

    def get_coupled_variables(self, variables):

        j_sei = variables[self.domain + " electrode sei interfacial current density"]
        L_sei = variables["Total " + self.domain.lower() + " electrode sei thickness"]
        c_scale = self.param.c_ec_0_dim
        # concentration of EC on graphite surface, base case = 1
        if self.domain == "Negative":
            C_ec = self.param.C_ec_n

        c_ec = pybamm.Scalar(1) + j_sei * L_sei * C_ec
        c_ec_av = pybamm.x_average(c_ec)
        variables.update(
            {
                self.domain + " electrode EC surface concentration": c_ec,
                self.domain
                + " electrode EC surface concentration [mol.m-3]": c_ec * c_scale,
                "X-averaged "
                + self.domain.lower()
                + " electrode EC surface concentration": c_ec_av,
                "X-averaged "
                + self.domain.lower()
                + " electrode EC surface concentration": c_ec_av * c_scale,
            }
        )
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
        L_sei = variables["Total " + domain + " sei thickness"]
        j_sei = variables[self.domain + " electrode sei interfacial current density"]

        if self.domain == "Negative":
            Gamma_SEI = self.param.Gamma_SEI_n

        self.rhs = {L_sei: -Gamma_SEI * j_sei / 2}

    def set_algebraic(self, variables):
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]
        j_sei = variables[self.domain + " electrode sei interfacial current density"]
        L_sei = variables["Total " + self.domain.lower() + " electrode sei thickness"]
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
        L_sei = variables["Total " + self.domain.lower() + " electrode sei thickness"]
        j_sei = variables[self.domain + " electrode sei interfacial current density"]

        L_sei_0 = pybamm.Scalar(1)
        j_sei_0 = pybamm.Scalar(0)

        self.initial_conditions = {L_sei: L_sei_0, j_sei: j_sei_0}
