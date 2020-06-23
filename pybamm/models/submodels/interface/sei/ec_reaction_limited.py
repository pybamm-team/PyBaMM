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
            "Total " + self.domain.lower() +
            " electrode sei thickness",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        j_sei = pybamm.Variable(
            self.domain + " electrode sei interfacial current density",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        L_sei_av = pybamm.x_average(L_sei)
        j_sei_av = pybamm.x_average(j_sei)

        L_scale = pybamm.sei_parameters.L_sei_0_dim
        # in this model the scale is identical to the intercalation current
        j_scale = pybamm.sei_parameters.j_scale_n
        R_sei_dim = pybamm.sei_parameters.R_sei_dimensional

        variables = {
            "Total " + self.domain.lower() +
            " electrode sei thickness": L_sei,
            "Total " + self.domain.lower() +
            " sei thickness [m]": L_sei * L_scale,
            "X-averaged total " + self.domain.lower() +
            " electrode sei thickness": L_sei_av,
            "X-averaged total " + self.domain.lower() +
            " electrode sei thickness [m]": L_sei_av * L_scale,
            self.domain +
            " electrode sei interfacial current density": j_sei,
            self.domain +
            " electrode scaled sei interfacial current density": j_sei,
            self.domain +
            " electrode sei interfacial current density [A.m-2]": j_sei * j_scale,
            "X-averaged " + self.domain.lower() +
            " electrode sei interfacial current density": j_sei_av,
            "X-averaged " + self.domain.lower() +
            " electrode sei interfacial current density [A.m-2]": j_sei_av * j_scale,
            "X-averaged " + self.domain.lower() +
            " electrode scaled sei interfacial current density": j_sei_av,
            "X-averaged " + self.domain.lower() +
            " electrode resistance [Ohm.m2]": L_sei_av * L_scale * R_sei_dim,
        }

        return variables

    def get_coupled_variables(self, variables):

        j_sei = variables[self.domain +
                          " electrode sei interfacial current density"]
        L_sei = variables["Total " + self.domain.lower() +
                          " electrode sei thickness"]
        C_ec = pybamm.sei_parameters.C_ec
        c_scale = pybamm.sei_parameters.c_ec_0_dim
        # concentration of EC on graphite surface, base case = 1
        c_ec = pybamm.Scalar(1) + j_sei * L_sei * C_ec
        c_ec_av = pybamm.x_average(c_ec)
        variables = {
            self.domain +
            " electrode EC surface concentration": c_ec,
            self.domain +
            " electrode EC surface concentration [mol.m-3]": c_ec * c_scale,
            "X-averaged " + self.domain.lower() +
            " electrode EC surface concentration": c_ec_av,
            "X-averaged " + self.domain.lower() +
            " electrode EC surface concentration": c_ec_av * c_scale,
        }
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
        j_sei = variables[self.domain +
                          " electrode sei interfacial current density"]

        C_sei_j = pybamm.sei_parameters.C_sei_j

        self.rhs = {L_sei: -j_sei * C_sei_j}

    def set_algebraic(self, variables):
        phi_s_n = variables[self.domain + " electrode potential"]
        phi_e_n = variables[self.domain + " electrolyte potential"]
        j_sei = variables[self.domain +
                          " electrode sei interfacial current density"]
        L_sei = variables["Total " + self.domain.lower() +
                          " electrode sei thickness"]
        c_ec = variables[self.domain + " electrode EC surface concentration"]

        # Look for current that contributes to the -IR drop
        # If we can't find the interfacial current density from the main reaction, j,
        # it's ok to fall back on the total interfacial current density, j_tot
        # This should only happen when the interface submodel is "InverseButlerVolmer"
        # in which case j = j_tot (uniform) anyway
        try:
            j = variables["Total " +
                          self.domain.lower() +
                          " electrode interfacial current density"]
        except KeyError:
            j = variables["X-averaged " + self.domain.lower()
                          + " electrode total interfacial current density"]
        C_sei_ec = pybamm.sei_parameters.C_sei_ec
        R_sei = pybamm.sei_parameters.R_sei

        # need to revise for thermal case

        self.algebraic = {j_sei: j_sei + C_sei_ec * c_ec * pybamm.exp(- 0.5 * (
            phi_s_n - phi_e_n - j * L_sei * R_sei))}

    def set_initial_conditions(self, variables):
        L_sei = variables["Total " + self.domain.lower() +
                          " electrode sei thickness"]
        j_sei = variables[self.domain +
                          " electrode sei interfacial current density"]

        L_sei_0 = pybamm.Scalar(1)
        j_sei_0 = pybamm.Scalar(0)

        self.initial_conditions = {L_sei: L_sei_0,
                                   j_sei: j_sei_0}
