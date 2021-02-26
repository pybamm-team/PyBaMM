#
# Class for composite electrolyte diffusion employing stefan-maxwell
#
import pybamm
from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class Composite(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Composite refers to composite model by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms
    extended : bool
        Whether to include feedback from the first-order terms

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, extended=False):
        super().__init__(param)
        self.extended = extended

    def get_fundamental_variables(self):
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p

        return self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)

    def get_coupled_variables(self, variables):

        tor_0 = variables["Leading-order electrolyte tortuosity"]
        eps = variables["Leading-order porosity"]
        c_e_0_av = variables["Leading-order x-averaged electrolyte concentration"]
        c_e = variables["Electrolyte concentration"]
        i_e = variables["Electrolyte current density"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]
        T_0 = variables["Leading-order cell temperature"]

        param = self.param

        N_e_diffusion = -tor_0 * param.D_e(c_e_0_av, T_0) * pybamm.grad(c_e)
        N_e_migration = param.C_e * param.t_plus(c_e, T_0) * i_e / param.gamma_e
        N_e_convection = param.C_e * c_e_0_av * v_box_0

        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))
        variables.update(self._get_total_concentration_electrolyte(c_e, eps))

        return variables

    def set_rhs(self, variables):
        """Composite reaction-diffusion with source terms from leading order."""

        param = self.param

        eps_0 = variables["Leading-order porosity"]
        deps_0_dt = variables["Leading-order porosity change"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        if self.extended is False:
            sum_s_j = variables[
                "Leading-order sum of electrolyte reaction source terms"
            ]
        elif self.extended == "distributed":
            sum_s_j = variables["Sum of electrolyte reaction source terms"]
        elif self.extended == "average":
            sum_s_j_n_av = variables[
                "Sum of x-averaged negative electrode electrolyte reaction source terms"
            ]
            sum_s_j_p_av = variables[
                "Sum of x-averaged positive electrode electrolyte reaction source terms"
            ]
            sum_s_j = pybamm.Concatenation(
                pybamm.PrimaryBroadcast(sum_s_j_n_av, "negative electrode"),
                pybamm.FullBroadcast(0, "separator", "current collector"),
                pybamm.PrimaryBroadcast(sum_s_j_p_av, "positive electrode"),
            )
        source_terms = sum_s_j / self.param.gamma_e

        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms - c_e * deps_0_dt)
        }

    def set_initial_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.initial_conditions = {c_e: self.param.c_e_init}

    def set_boundary_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
