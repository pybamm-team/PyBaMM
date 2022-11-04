#
# Class for composite electrolyte diffusion employing stefan-maxwell
#
import pybamm
import numpy as np
from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class Composite(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Composite refers to composite model by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    extended : bool
        Whether to include feedback from the first-order terms

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, extended=False):
        super().__init__(param)
        self.extended = extended

    def get_fundamental_variables(self):
        c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize().split()[0]
            c_e_k = pybamm.Variable(
                f"{Domain} electrolyte concentration [mol.m-3]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, np.inf),
            )
            c_e_k.print_name = f"c_e_{domain[0]}"
            c_e_dict[domain] = c_e_k

        variables = self._get_standard_concentration_variables(c_e_dict)

        return variables

    def get_coupled_variables(self, variables):

        tor_0 = variables["Leading-order electrolyte transport efficiency"]
        eps = variables["Leading-order porosity"]
        c_e_0_av = variables[
            "Leading-order x-averaged electrolyte concentration [mol.m-3]"
        ]
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        i_e = variables["Electrolyte current density [A.m-2]"]
        v_box_0 = variables["Leading-order volume-averaged velocity [m.s-1]"]
        T_0 = variables["Leading-order cell temperature [K]"]

        param = self.param

        N_e_diffusion = -tor_0 * param.D_e(c_e_0_av, T_0) * pybamm.grad(c_e)
        N_e_migration = param.t_plus(c_e, T_0) * i_e / param.F
        N_e_convection = c_e_0_av * v_box_0

        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))
        variables.update(self._get_total_concentration_electrolyte(eps * c_e))

        return variables

    def set_rhs(self, variables):
        """Composite reaction-diffusion with source terms from leading order."""

        param = self.param

        eps_0 = variables["Leading-order porosity"]
        deps_0_dt = variables["Leading-order porosity change [s-1]"]
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        N_e = variables["Electrolyte flux [mol.m-2.s-1]"]
        if self.extended is False:
            sum_s_a_j = variables[
                "Leading-order sum of electrolyte reaction source terms [A.m-3]"
            ]
        elif self.extended == "distributed":
            sum_s_a_j = variables["Sum of electrolyte reaction source terms [A.m-3]"]
        elif self.extended == "average":
            sum_s_a_j_n_av = variables[
                "Sum of x-averaged negative electrode electrolyte "
                "reaction source terms [A.m-3]"
            ]
            sum_s_a_j_p_av = variables[
                "Sum of x-averaged positive electrode electrolyte "
                "reaction source terms [A.m-3]"
            ]
            sum_s_a_j = pybamm.concatenation(
                pybamm.PrimaryBroadcast(sum_s_a_j_n_av, "negative electrode"),
                pybamm.FullBroadcast(0, "separator", "current collector"),
                pybamm.PrimaryBroadcast(sum_s_a_j_p_av, "positive electrode"),
            )
        source_terms = sum_s_a_j / param.F

        self.rhs = {
            c_e: (1 / eps_0) * (-pybamm.div(N_e) + source_terms - c_e * deps_0_dt)
        }

    def set_initial_conditions(self, variables):

        c_e = variables["Electrolyte concentration [mol.m-3]"]

        self.initial_conditions = {c_e: self.param.c_e_init}

    def set_boundary_conditions(self, variables):

        c_e = variables["Electrolyte concentration [mol.m-3]"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
