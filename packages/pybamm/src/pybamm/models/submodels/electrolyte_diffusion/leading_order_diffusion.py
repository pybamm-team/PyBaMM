#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import numpy as np

import pybamm

from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class LeadingOrder(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Leading refers to leading order
    of asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        whole_cell = self.options.whole_cell_domains

        def coupled_average(name):
            return self.coupled_variable(name, domain="current collector")

        T_av = coupled_average("X-averaged cell temperature [K]")
        eps_n_av = coupled_average("X-averaged negative electrode porosity")
        eps_s_av = coupled_average("X-averaged separator porosity")
        eps_p_av = coupled_average("X-averaged positive electrode porosity")
        deps_n_dt_av = coupled_average(
            "X-averaged negative electrode porosity change [s-1]"
        )
        deps_p_dt_av = coupled_average(
            "X-averaged positive electrode porosity change [s-1]"
        )
        div_Vbox_s_av = coupled_average(
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]"
        )
        sum_a_j_n_0, sum_a_j_p_0 = (
            coupled_average(
                f"Sum of x-averaged {domain} electrode volumetric "
                "interfacial current densities [A.m-3]"
            )
            for domain in ["negative", "positive"]
        )
        sum_s_j_n_0, sum_s_j_p_0 = (
            coupled_average(
                f"Sum of x-averaged {domain} electrode electrolyte "
                "reaction source terms [A.m-3]"
            )
            for domain in ["negative", "positive"]
        )
        eps = self.coupled_variable(
            "Porosity",
            domain=whole_cell,
            auxiliary_domains={"secondary": "current collector"},
        )

        c_e_av = pybamm.Variable(
            "X-averaged electrolyte concentration [mol.m-3]",
            domain="current collector",
            bounds=(0, np.inf),
        )
        c_e_av.print_name = "c_e_av"

        c_e_dict = {
            domain: pybamm.PrimaryBroadcast(c_e_av, domain) for domain in whole_cell
        }
        variables = self._get_standard_concentration_variables(c_e_dict)

        N_e = pybamm.FullBroadcastToEdges(0, whole_cell, "current collector")
        variables.update(self._get_standard_flux_variables(N_e))

        c_e = variables["Electrolyte concentration [mol.m-3]"]
        variables.update(self._get_total_concentration_electrolyte(eps * c_e))
        self.variables.update(variables)

        L_n = self.param.n.L
        L_s = self.param.s.L
        L_p = self.param.p.L
        t_plus = self.param.t_plus(c_e_av, T_av)
        source_terms = (
            L_n * (sum_s_j_n_0 - t_plus * sum_a_j_n_0)
            + L_p * (sum_s_j_p_0 - t_plus * sum_a_j_p_0)
        ) / self.param.F

        self.rhs = {
            c_e_av: 1
            / (L_n * eps_n_av + L_s * eps_s_av + L_p * eps_p_av)
            * (
                source_terms
                - c_e_av * (L_n * deps_n_dt_av + L_p * deps_p_dt_av)
                - c_e_av * L_s * div_Vbox_s_av
            )
        }
        self.initial_conditions = {c_e_av: self.param.c_e_init}
