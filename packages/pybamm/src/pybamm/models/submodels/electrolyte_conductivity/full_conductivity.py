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
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def build(self):
        whole_cell = self.options.whole_cell_domains
        T = self._coupled_domain_variable("Cell temperature [K]", whole_cell)
        tor = self._coupled_domain_variable(
            "Electrolyte transport efficiency", whole_cell
        )
        c_e = self._coupled_domain_variable(
            "Electrolyte concentration [mol.m-3]", whole_cell
        )
        sum_a_j = self._coupled_domain_variable(
            "Sum of volumetric interfacial current densities [A.m-3]", whole_cell
        )

        phi_e_dict = {}
        for domain in whole_cell:
            phi_e_k = pybamm.Variable(
                f"{domain.split()[0].capitalize()} electrolyte potential [V]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                reference=-self.param.n.prim.U_init,
            )
            phi_e_k.print_name = f"phi_e_{domain[0]}"
            phi_e_dict[domain] = phi_e_k
        variables = self._get_standard_potential_variables(phi_e_dict)
        phi_e = variables["Electrolyte potential [V]"]

        i_e = (self.param.kappa_e(c_e, T) * tor) * (
            self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        i_e.print_name = "i_e"

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_electrolyte_overpotentials(phi_e_dict))
        self.variables.update(variables)

        # multiply by Lx**2 to improve conditioning
        self.algebraic = {phi_e: self.param.L_x**2 * (pybamm.div(i_e) - sum_a_j)}
        self.initial_conditions = {phi_e: -self.param.n.prim.U_init}
        self.boundary_conditions = self._get_electrolyte_potential_boundary_conditions(
            phi_e
        )
