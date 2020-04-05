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

    def __init__(self, param, reactions, extended=False):
        super().__init__(param, reactions)
        self.extended = extended

    def get_fundamental_variables(self):
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p

        return self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)

    def get_coupled_variables(self, variables):

        tor_0 = variables["Leading-order electrolyte tortuosity"]
        c_e_0_av = variables["Leading-order x-averaged electrolyte concentration"]
        c_e = variables["Electrolyte concentration"]
        # i_e = variables["Electrolyte current density"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]
        T_0 = variables["Leading-order cell temperature"]

        param = self.param

        N_e_diffusion = -tor_0 * param.D_e(c_e_0_av, T_0) * pybamm.grad(c_e)

        N_e = N_e_diffusion + param.C_e * c_e_0_av * v_box_0

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):
        "Composite reaction-diffusion with source terms from leading order"

        param = self.param

        eps_0 = variables["Leading-order porosity"]
        deps_0_dt = variables["Leading-order porosity change"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        if self.extended is False:
            source_terms_0 = self._get_source_terms_leading_order(variables)
        elif self.extended == "distributed":
            source_terms_0 = self._get_source_terms_first_order(variables)
        elif self.extended == "average":
            source_terms_0 = self._get_source_terms_first_order_average(variables)

        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms_0 - c_e * deps_0_dt)
        }

    def _get_source_terms_leading_order(self, variables):
        param = self.param
        c_e_n = variables["Negative electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]

        return sum(
            pybamm.Concatenation(
                (reaction["Negative"]["s"] - param.t_plus(c_e_n))
                * variables["Leading-order " + reaction["Negative"]["aj"].lower()],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                (reaction["Positive"]["s"] - param.t_plus(c_e_p))
                * variables["Leading-order " + reaction["Positive"]["aj"].lower()],
            )
            / self.param.gamma_e
            for reaction in self.reactions.values()
        )

    def _get_source_terms_first_order(self, variables):
        param = self.param
        c_e_n = variables["Negative electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]

        return sum(
            pybamm.Concatenation(
                (reaction["Negative"]["s"] - param.t_plus(c_e_n))
                * variables[reaction["Negative"]["aj"]],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                (reaction["Positive"]["s"] - param.t_plus(c_e_p))
                * variables[reaction["Positive"]["aj"]],
            )
            / self.param.gamma_e
            for reaction in self.reactions.values()
        )

    def _get_source_terms_first_order_average(self, variables):
        first_order_average = sum(
            (
                reaction["Negative"]["s"]
                * variables[
                    "First-order x-averaged " + reaction["Negative"]["aj"].lower()
                ]
                + reaction["Positive"]["s"]
                * variables[
                    "First-order x-averaged " + reaction["Positive"]["aj"].lower()
                ]
            )
            / self.param.gamma_e
            for reaction in self.reactions.values()
        )

        return self._get_source_terms_leading_order(
            variables
        ) + self.param.C_e * pybamm.PrimaryBroadcast(
            first_order_average,
            ["negative electrode", "separator", "positive electrode"],
        )

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
