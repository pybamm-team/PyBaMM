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

    def get_fundamental_variables(self):
        phi_e_dict = {}
        variables = {}
        for domain in self.options.whole_cell_domains:
            Dom = domain.capitalize().split()[0]
            name = f"{Dom} electrolyte potential [V]"
            phi_e_k = pybamm.Variable(
                name,
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                reference=-self.param.n.prim.U_init,
            )
            phi_e_k.print_name = f"phi_e_{domain[0]}"
            phi_e_dict[domain] = phi_e_k

        variables["Electrolyte potential [V]"] = pybamm.concatenation(
            *phi_e_dict.values()
        )

        variables.update(self._get_standard_potential_variables(phi_e_dict))

        return variables

    def get_coupled_variables(self, variables):
        if self.options.electrode_types["negative"] == "planar":
            i_e_n = None
        else:
            T_n = variables["Negative electrode temperature [K]"]
            tor_n = variables["Negative electrolyte transport efficiency"]
            c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
            phi_e_n = variables["Negative electrolyte potential [V]"]
            i_e_n = (self.param.kappa_e(c_e_n, T_n) * tor_n) * (
                self.param.chiRT_over_Fc(c_e_n, T_n) * pybamm.grad(c_e_n)
                - pybamm.grad(phi_e_n)
            )

        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]

        tor_s = variables["Separator electrolyte transport efficiency"]
        tor_p = variables["Positive electrolyte transport efficiency"]

        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        phi_e_s = variables["Separator electrolyte potential [V]"]
        phi_e_p = variables["Positive electrolyte potential [V]"]

        i_e_s = (self.param.kappa_e(c_e_s, T_s) * tor_s) * (
            self.param.chiRT_over_Fc(c_e_s, T_s) * pybamm.grad(c_e_s)
            - pybamm.grad(phi_e_s)
        )

        i_e_p = (self.param.kappa_e(c_e_p, T_p) * tor_p) * (
            self.param.chiRT_over_Fc(c_e_p, T_p) * pybamm.grad(c_e_p)
            - pybamm.grad(phi_e_p)
        )

        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)
        # Override print_name
        i_e.print_name = "i_e"

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_electrolyte_overpotentials(variables))

        return variables

    def set_algebraic(self, variables):
        # phi_e = variables["Electrolyte potential [V]"]
        phi_e_n = variables["Negative electrolyte potential [V]"]
        phi_e_s = variables["Separator electrolyte potential [V]"]
        phi_e_p = variables["Positive electrolyte potential [V]"]

        i_e = variables["Electrolyte current density [A.m-2]"]
        if self.options.electrode_types["negative"] == "porous":
            i_e_n, i_e_s, i_e_p = i_e.orphans
            # Variable summing all of the interfacial current densities
            sum_a_j_n = variables[
                "Sum of negative electrode volumetric interfacial current densities [A.m-3]"
            ]
            sum_a_j_s = pybamm.FullBroadcast(0, "separator")
            sum_a_j_p = variables[
                "Sum of positive electrode volumetric interfacial current densities [A.m-3]"
            ]

            # Override print_name
            sum_a_j_n.print_name = "aj_n"
            sum_a_j_s.print_name = "aj_s"
            sum_a_j_p.print_name = "aj_p"

            # multiply by Lx**2 to improve conditioning
            self.algebraic = {
                phi_e_n: self.param.L_x**2 * (pybamm.div(i_e_n) - sum_a_j_n),
                phi_e_s: self.param.L_x**2 * (pybamm.div(i_e_s) - sum_a_j_s),
                phi_e_p: self.param.L_x**2 * (pybamm.div(i_e_p) - sum_a_j_p),
            }
        else:
            i_e_s, i_e_p = i_e.orphans
            # Variable summing all of the interfacial current densities
            sum_a_j_s = pybamm.FullBroadcast(0, "separator")
            sum_a_j_p = variables[
                "Sum of positive electrode volumetric interfacial current densities [A.m-3]"
            ]

            # Override print_name
            sum_a_j_s.print_name = "aj_s"
            sum_a_j_p.print_name = "aj_p"

            # multiply by Lx**2 to improve conditioning
            self.algebraic = {
                phi_e_s: self.param.L_x**2 * (pybamm.div(i_e_s) - sum_a_j_s),
                phi_e_p: self.param.L_x**2 * (pybamm.div(i_e_p) - sum_a_j_p),
            }

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential [V]"]
        if self.options.electrode_types["negative"] == "porous":
            phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
            self.initial_conditions = {
                phi_e_n: -self.param.n.prim.U_init,
                phi_e_s: -self.param.n.prim.U_init,
                phi_e_p: -self.param.n.prim.U_init,
            }
        else:
            phi_e_s, phi_e_p = phi_e.orphans
            self.initial_conditions = {
                phi_e_s: -self.param.n.prim.U_init,
                phi_e_p: -self.param.n.prim.U_init,
            }
