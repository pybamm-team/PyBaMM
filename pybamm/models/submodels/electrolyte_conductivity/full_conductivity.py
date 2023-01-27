#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Full(BaseElectrolyteConductivity):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        phi_e_dict = {}
        for domain in self.options.whole_cell_domains:
            phi_e_k = pybamm.Variable(
                f"{domain.capitalize().split()[0]} electrolyte potential",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
            phi_e_k.print_name = f"phi_e_{domain[0]}"
            phi_e_dict[domain] = phi_e_k

        variables = self._get_standard_potential_variables(phi_e_dict)
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        T = variables["Cell temperature"]
        tor = variables["Electrolyte transport efficiency"]
        c_e = variables["Electrolyte concentration"]
        phi_e = variables["Electrolyte potential"]

        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )

        # Override print_name
        i_e.print_name = "i_e"

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_electrolyte_overpotentials(variables))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]

        # Variable summing all of the interfacial current densities
        sum_a_j = variables["Sum of volumetric interfacial current densities"]

        # Override print_name
        sum_a_j.print_name = "aj"

        self.algebraic = {phi_e: pybamm.div(i_e) - sum_a_j}

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        self.initial_conditions = {phi_e: -self.param.n.prim.U_init}
