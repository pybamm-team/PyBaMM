#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
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
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        c_e_av = pybamm.standard_variables.c_e_av
        c_e_n = pybamm.PrimaryBroadcast(c_e_av, ["negative electrode"])
        c_e_s = pybamm.PrimaryBroadcast(c_e_av, ["separator"])
        c_e_p = pybamm.PrimaryBroadcast(c_e_av, ["positive electrode"])

        return self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)

    def get_coupled_variables(self, variables):

        N_e = pybamm.FullBroadcastToEdges(
            0,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        c_e_av = variables["X-averaged electrolyte concentration"]

        eps_n_av = variables["X-averaged negative electrode porosity"]
        eps_s_av = variables["X-averaged separator porosity"]
        eps_p_av = variables["X-averaged positive electrode porosity"]

        deps_n_dt_av = variables["X-averaged negative electrode porosity change"]
        deps_p_dt_av = variables["X-averaged positive electrode porosity change"]

        div_Vbox_s_av = variables[
            "X-averaged separator transverse volume-averaged acceleration"
        ]

        sum_j_n_0 = variables[
            "Sum of x-averaged negative electrode interfacial current densities"
        ]
        sum_j_p_0 = variables[
            "Sum of x-averaged positive electrode interfacial current densities"
        ]
        sum_s_j_n_0 = variables[
            "Sum of x-averaged negative electrode electrolyte reaction source terms"
        ]
        sum_s_j_p_0 = variables[
            "Sum of x-averaged positive electrode electrolyte reaction source terms"
        ]
        source_terms = (
            param.l_n * (sum_s_j_n_0 - param.t_plus(c_e_av) * sum_j_n_0)
            + param.l_p * (sum_s_j_p_0 - param.t_plus(c_e_av) * sum_j_p_0)
        ) / param.gamma_e

        self.rhs = {
            c_e_av: 1
            / (param.l_n * eps_n_av + param.l_s * eps_s_av + param.l_p * eps_p_av)
            * (
                source_terms
                - c_e_av * (param.l_n * deps_n_dt_av + param.l_p * deps_p_dt_av)
                - c_e_av * param.l_s * div_Vbox_s_av
            )
        }

    def set_initial_conditions(self, variables):
        c_e = variables["X-averaged electrolyte concentration"]
        self.initial_conditions = {c_e: self.param.c_e_init}
