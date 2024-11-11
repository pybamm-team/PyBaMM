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
                self.param.chiRT_over_Fc(c_e_n, T_n) * pybamm.grad(c_e_n) - pybamm.grad(phi_e_n)
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
            self.param.chiRT_over_Fc(c_e_s, T_s) * pybamm.grad(c_e_s) - pybamm.grad(phi_e_s)
        )

        i_e_p = (self.param.kappa_e(c_e_p, T_p) * tor_p) * (
            self.param.chiRT_over_Fc(c_e_p, T_p) * pybamm.grad(c_e_p) - pybamm.grad(phi_e_p)
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
            sum_a_j_s = pybamm.PrimaryBroadcast(0, "separator")
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
            sum_a_j_s = pybamm.PrimaryBroadcast(0, "separator")
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

    def set_boundary_conditions(self, variables):
        phi_e = variables["Electrolyte potential [V]"]
        phi_e_s = variables["Separator electrolyte potential [V]"]
        phi_e_p = variables["Positive electrolyte potential [V]"]

        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]

        tor_s = variables["Separator electrolyte transport efficiency"]
        tor_p = variables["Positive electrolyte transport efficiency"]
        
        
        kappa_tor_s_left = pybamm.boundary_value(self.param.kappa_e(c_e_s, T_s) * tor_s, "left")
        kappa_tor_s_right = pybamm.boundary_value(self.param.kappa_e(c_e_s, T_s) * tor_s, "right")
        kappa_tor_p_left = pybamm.boundary_value(self.param.kappa_e(c_e_p, T_p) * tor_p, "left")

        chi_rt_over_fc_s_left = pybamm.boundary_value(self.param.chiRT_over_Fc(c_e_s, T_s), "left")
        chi_rt_over_fc_s_right = pybamm.boundary_value(self.param.chiRT_over_Fc(c_e_s, T_s), "right")
        chi_rt_over_fc_p_left = pybamm.boundary_value(self.param.chiRT_over_Fc(c_e_p, T_p), "left")

        grad_c_e_s_left = pybamm.boundary_gradient(c_e_s, "left")
        grad_c_e_s_right = pybamm.boundary_gradient(c_e_s, "right")
        grad_c_e_p_left = pybamm.boundary_gradient(c_e_p, "left")

        grad_phi_e_s_left = pybamm.boundary_gradient(phi_e_s, "left")
        grad_phi_e_s_right = pybamm.boundary_gradient(phi_e_s, "right")
        grad_phi_e_p_left = pybamm.boundary_gradient(phi_e_p, "left")
        
        # separator / positive electrode
        phi_e_s_rbc = chi_rt_over_fc_s_right * grad_c_e_s_right - ((kappa_tor_p_left / kappa_tor_s_right) * (chi_rt_over_fc_p_left * grad_c_e_p_left - grad_phi_e_p_left))
        phi_e_p_lbc = chi_rt_over_fc_p_left * grad_c_e_p_left - ((kappa_tor_s_right / kappa_tor_p_left) * (chi_rt_over_fc_s_right * grad_c_e_s_right - grad_phi_e_s_right))
        
        # positive electrode / right
        phi_e_p_rbc = pybamm.Scalar(0)
        if self.options.electrode_types["negative"] == "porous":
            phi_e_n = variables["Negative electrolyte potential [V]"]
            c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
            T_n = variables["Negative electrode temperature [K]"]
            tor_n = variables["Negative electrolyte transport efficiency"]
            kappa_tor_n_right = pybamm.boundary_value(self.param.kappa_e(c_e_n, T_n) * tor_n, "right")
            chi_rt_over_fc_n_right = pybamm.boundary_value(self.param.chiRT_over_Fc(c_e_n, T_n), "right")
            grad_c_e_n_right = pybamm.boundary_gradient(c_e_n, "right")
            phi_e_n_rbc = chi_rt_over_fc_n_right * grad_c_e_n_right - ((kappa_tor_s_left / kappa_tor_n_right) * (chi_rt_over_fc_s_left * grad_c_e_s_left - grad_phi_e_s_left))
            phi_e_n_lbc = pybamm.Scalar(0)
            self.boundary_conditions[phi_e_n] = {
                "left": (phi_e_n_lbc, "Neumann"),
                "right": (phi_e_n_rbc, "Neumann"),
            }
            self.boundary_conditions[phi_e_s] = {
                "left": (pybamm.boundary_value(phi_e_n, "right"), "Dirichlet"),
                "right": (pybamm.boundary_value(phi_e_p, "left"), "Dirichlet"),
            }
        else:
            phi_e_ref = variables["Lithium metal interface electrolyte potential [V]"]
            lbc = (phi_e_ref, "Dirichlet")
            self.boundary_conditions[phi_e_s] = {
                "left": (phi_e_ref, "Dirichlet"),
                "right": (pybamm.boundary_value(phi_e_p, "left"), "Dirichlet"),
            }
        self.boundary_conditions[phi_e_p] = {
            "left": (phi_e_p_lbc, "Neumann"),
            "right": (phi_e_p_rbc, "Neumann"),
        }

