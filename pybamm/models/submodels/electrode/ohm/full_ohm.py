#
# Full model of electrode employing Ohm's law
#
import pybamm
from .base_ohm import BaseModel


class Full(BaseModel):
    """Full model of electrode employing Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, domain, reactions):
        super().__init__(param, domain, reactions)

    def get_fundamental_variables(self):

        if self.domain == "Negative":
            phi_s = pybamm.standard_variables.phi_s_n
        elif self.domain == "Positive":
            phi_s = pybamm.standard_variables.phi_s_p

        variables = self._get_standard_potential_variables(phi_s)

        return variables

    def get_coupled_variables(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        tor = variables[self.domain + " electrode tortuosity"]

        if self.domain == "Negative":
            sigma = self.param.sigma_n
        elif self.domain == "Positive":
            sigma = self.param.sigma_p

        sigma_eff = sigma * tor
        i_s = -sigma_eff * pybamm.grad(phi_s)

        variables.update({self.domain + " electrode effective conductivity": sigma_eff})

        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_variables(variables))

        return variables

    def set_algebraic(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        i_s = variables[self.domain + " electrode current density"]
        sum_j = sum(
            variables[reaction[self.domain]["aj"]]
            for reaction in self.reactions.values()
        )

        self.algebraic[phi_s] = pybamm.div(i_s) + sum_j

    def set_boundary_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        phi_s_cn = variables["Negative current collector potential"]
        tor = variables[self.domain + " electrode tortuosity"]
        i_boundary_cc = variables["Current collector current density"]

        if self.domain == "Negative":
            lbc = (phi_s_cn, "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.sigma_p * tor
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    def set_initial_conditions(self, variables):
        l_n = self.param.l_n
        x_n = pybamm.standard_spatial_vars.x_n
        l_p = self.param.l_p
        x_p = pybamm.standard_spatial_vars.x_p
        phi_s = variables[self.domain + " electrode potential"]
        T_init = self.param.T_init
        i_init = pybamm.PrimaryBroadcast(
            self.param.current_with_time, "current collector"
        )
        tor = variables[self.domain + " electrode tortuosity"]

        if self.domain == "Negative":
            sigma_eff = self.param.sigma_n * tor

            phi_s_init = (i_init / sigma_eff) * (x_n * (x_n - 2 * l_n) / (2 * l_n))
        elif self.domain == "Positive":

            U = self.param.U_p(self.param.c_p_init, T_init) - self.param.U_n(
                self.param.c_n_init, T_init
            )
            j0_n = (
                self.param.m_n(T_init)
                / self.param.C_r_n
                * self.param.c_e_init ** (1 / 2)
                * self.param.c_n_init ** (1 / 2)
                * (1 - self.param.c_n_init) ** (1 / 2)
            )
            j0_p = (
                self.param.gamma_p
                * self.param.m_p(T_init)
                / self.param.C_r_p
                * self.param.c_e_init ** (1 / 2)
                * self.param.c_p_init ** (1 / 2)
                * (1 - self.param.c_p_init) ** (1 / 2)
            )
            eta_r = -(
                2 * (1 + self.param.Theta * T_init) / self.param.ne_p
            ) * pybamm.arcsinh(i_init / (2 * j0_p * l_p)) - (
                2 * (1 + self.param.Theta * T_init) / self.param.ne_n
            ) * pybamm.arcsinh(
                i_init / (2 * j0_n * l_n)
            )

            sigma_eff = self.param.sigma_p * tor

            phi_s_init = (
                U + eta_r - (i_init / sigma_eff) * (x_p + (x_p - 1) ** 2 / (2 * l_p))
            )

        self.initial_conditions[phi_s] = phi_s_init
