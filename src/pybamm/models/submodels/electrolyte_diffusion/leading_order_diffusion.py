#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import pybamm
import numpy as np

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
        c_e_av = pybamm.Variable(
            "X-averaged electrolyte concentration [mol.m-3]",
            domain="current collector",
            bounds=(0, np.inf),
        )
        c_e_av.print_name = "c_e_av"

        c_e_dict = {
            domain: pybamm.PrimaryBroadcast(c_e_av, domain)
            for domain in self.options.whole_cell_domains
        }

        variables = self._get_standard_concentration_variables(c_e_dict)

        N_e = pybamm.FullBroadcastToEdges(
            0, self.options.whole_cell_domains, "current collector"
        )
        variables.update(self._get_standard_flux_variables(N_e))

        c_e = variables["Electrolyte concentration [mol.m-3]"]

        eps = pybamm.CoupledVariable(
            "Porosity",
            domains={
                "primary": self.options.whole_cell_domains,
                "secondary": "current collector",
            },
        )
        self.coupled_variables.update({eps.name: eps})

        variables.update(self._get_total_concentration_electrolyte(eps * c_e))

        c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]

        T_av = pybamm.CoupledVariable(
            "X-averaged cell temperature [K]", domain="current collector"
        )
        self.coupled_variables.update({T_av.name: T_av})

        eps_n_av = pybamm.CoupledVariable(
            "X-averaged negative electrode porosity", domain="current collector"
        )
        self.coupled_variables.update({eps_n_av.name: eps_n_av})
        eps_s_av = pybamm.CoupledVariable(
            "X-averaged separator porosity", domain="current collector"
        )
        self.coupled_variables.update({eps_s_av.name: eps_s_av})
        eps_p_av = pybamm.CoupledVariable(
            "X-averaged positive electrode porosity", domain="current collector"
        )
        self.coupled_variables.update({eps_p_av.name: eps_p_av})
        deps_n_dt_av = pybamm.CoupledVariable(
            "X-averaged negative electrode porosity change [s-1]",
            domain="current collector",
        )
        self.coupled_variables.update({deps_n_dt_av.name: deps_n_dt_av})
        deps_p_dt_av = pybamm.CoupledVariable(
            "X-averaged positive electrode porosity change [s-1]",
            domain="current collector",
        )
        self.coupled_variables.update({deps_p_dt_av.name: deps_p_dt_av})
        div_Vbox_s_av = pybamm.CoupledVariable(
            "X-averaged separator transverse volume-averaged acceleration [m.s-2]",
            domain="current collector",
        )
        self.coupled_variables.update({div_Vbox_s_av.name: div_Vbox_s_av})

        sum_a_j_n_0 = pybamm.CoupledVariable(
            "Sum of x-averaged negative electrode volumetric "
            "interfacial current densities [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_a_j_n_0.name: sum_a_j_n_0})
        sum_a_j_p_0 = pybamm.CoupledVariable(
            "Sum of x-averaged positive electrode volumetric "
            "interfacial current densities [A.m-3]"
        )
        self.coupled_variables.update({sum_a_j_p_0.name: sum_a_j_p_0})
        sum_s_j_n_0 = pybamm.CoupledVariable(
            "Sum of x-averaged negative electrode electrolyte "
            "reaction source terms [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_s_j_n_0.name: sum_s_j_n_0})
        sum_s_j_p_0 = pybamm.CoupledVariable(
            "Sum of x-averaged positive electrode electrolyte "
            "reaction source terms [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_s_j_p_0.name: sum_s_j_p_0})
        source_terms = (
            self.param.n.L
            * (sum_s_j_n_0 - self.param.t_plus(c_e_av, T_av) * sum_a_j_n_0)
            + self.param.p.L
            * (sum_s_j_p_0 - self.param.t_plus(c_e_av, T_av) * sum_a_j_p_0)
        ) / self.param.F

        self.rhs = {
            c_e_av: 1
            / (
                self.param.n.L * eps_n_av
                + self.param.s.L * eps_s_av
                + self.param.p.L * eps_p_av
            )
            * (
                source_terms
                - c_e_av
                * (self.param.n.L * deps_n_dt_av + self.param.p.L * deps_p_dt_av)
                - c_e_av * self.param.s.L * div_Vbox_s_av
            )
        }
        c_e = variables["X-averaged electrolyte concentration [mol.m-3]"]
        self.initial_conditions = {c_e: self.param.c_e_init}
        self.variables.update(variables)
