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

        variables = self._get_standard_thickness_variables(L_inner, L_outer)
        variables.update(self._get_standard_concentration_variables(variables))

        return variables

    def get_coupled_variables(self, variables):
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]
        L_sei = variables["Outer " + self.domain.lower() + " electrode SEI thickness"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        if self.domain + " electrode interfacial current density" in variables:
            j = variables[self.domain + " electrode interfacial current density"]
        else:
            j = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode total interfacial current density"
            ]

        if self.domain == "Negative":
            C_sei_ec = self.param.C_sei_ec_n
            R_sei = self.param.R_sei_n
            C_ec = self.param.C_ec_n

        # we have a linear system for j_sei and c_ec
        #  c_ec = 1 + j_sei * L_sei * C_ec
        #  j_sei = - C_sei_ec * c_ec * exp()
        # so
        #  j_sei = - C_sei_ec * exp() - j_sei * L_sei * C_ec * C_sei_ec * exp()
        # so
        #  j_sei = -C_sei_ec * exp() / (1 + L_sei * C_ec * C_sei_ec * exp())
        #  c_ec = 1 / (1 + L_sei * C_ec * C_sei_ec * exp())
        # need to revise for thermal case
        C_sei_exp = C_sei_ec * pybamm.exp(
            -0.5 * (phi_s_n - phi_e_n - j * L_sei * R_sei)
        )
        j_sei = -C_sei_exp / (1 + L_sei * C_ec * C_sei_exp)
        c_ec = 1 / (1 + L_sei * C_ec * C_sei_exp)

        j_inner = pybamm.FullBroadcast(
            0, self.domain.lower() + " electrode", "current collector"
        )
        j_outer = j_sei

        variables.update(self._get_standard_reaction_variables(j_inner, j_outer))

        # Get variables related to the concentration
        c_ec_av = pybamm.x_average(c_ec)
        c_ec_scale = self.param.c_ec_0_dim

        variables.update(
            {
                self.domain + " electrode EC surface concentration": c_ec,
                self.domain
                + " electrode EC surface concentration [mol.m-3]": c_ec * c_ec_scale,
                "X-averaged "
                + self.domain.lower()
                + " electrode EC surface concentration": c_ec_av,
                "X-averaged "
                + self.domain.lower()
                + " electrode EC surface concentration [mol.m-3]": c_ec_av * c_ec_scale,
            }
        )

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
        L_sei = variables["Outer " + domain + " SEI thickness"]
        j_sei = variables["Outer " + domain + " SEI interfacial current density"]

        if self.domain == "Negative":
            Gamma_SEI = self.param.Gamma_SEI_n

        self.rhs = {L_sei: -Gamma_SEI * j_sei / 2}

    def set_initial_conditions(self, variables):
        L_sei = variables["Outer " + self.domain.lower() + " electrode SEI thickness"]
        L_sei_0 = pybamm.Scalar(1)

        self.initial_conditions = {L_sei: L_sei_0}
