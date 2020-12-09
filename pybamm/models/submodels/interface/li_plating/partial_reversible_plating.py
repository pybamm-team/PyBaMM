#
# Class for partially reversible Li plating
#
import pybamm
from .base_plating import BasePlating


class PartialReversiblePlating(BasePlating):
    """Base class for reversible Li plating.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    **Extends:** :class:`pybamm.li_plating.BasePlating`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        c_plated_Li = pybamm.Variable(
            "Plated Li concentration",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        c_dead_Li = pybamm.Variable(
            "Dead Li concentration",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        variables = self._get_standard_concentration_variables(c_plated_Li,c_dead_Li)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        phi_s_n = variables[f"{self.domain} electrode potential"]
        phi_e_n = variables[f"{self.domain} electrolyte potential"]
        c_e_n = variables[f"{self.domain} electrolyte concentration"]
        c_plated_Li = variables[f"{self.domain} electrode Li plating concentration"]
        C_plating = param.C_plating
        phi_ref = param.U_n_ref / param.potential_scale

        if f"{self.domain} electrode sei film overpotential" in variables:
            eta_sei = variables[f"{self.domain} electrode sei film overpotential"]
        else:
            eta_sei = pybamm.Scalar(0)

        # need to revise for thermal case
        j_stripping = (1 / C_plating) * (
            c_plated_Li * pybamm.exp(0.5 * (phi_s_n - phi_e_n + phi_ref + eta_sei))
            - c_e_n * pybamm.exp(-0.5 * (phi_s_n - phi_e_n + phi_ref + eta_sei))
        )

        variables.update(self._get_standard_reaction_variables(j_stripping))

        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode Li plating interfacial current density" in variables
            and "Positive electrode Li plating interfacial current density" in variables
            and "Li plating interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables

    def set_rhs(self, variables):
        c_plated_Li = variables[f"{self.domain} electrode Li plating concentration"]
        j_stripping = variables[
            f"{self.domain} electrode Li plating " f"interfacial current density"
        ]
        c_dead_Li = variables[f"{self.domain} electrode dead Li concentration"]
        Gamma_plating = self.param.Gamma_plating
        SEIplating1 = self.param.SEIplating1

        self.rhs = {
            c_plated_Li: -Gamma_plating * j_stripping - SEIplating1 * c_plated_Li,
            c_dead_Li: SEIplating1 * c_plated_Li
        }

    def set_initial_conditions(self, variables):
        c_plated_Li = variables[f"{self.domain} electrode Li plating concentration"]
        c_plated_Li_0 = self.param.c_plated_Li_0
        c_dead_Li = variables[f"{self.domain} electrode dead Li concentration"]
        zero = pybamm.Scalar(0)

        self.initial_conditions = {c_plated_Li: c_plated_Li_0, c_dead_Li: zero}
