#
# Class for composite electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .full_stefan_maxwell_diffusion import Full


class Composite(Full):
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

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.Full`
    """

    def __init__(self, param, reactions, extended=False):
        super().__init__(param, reactions)
        self.extended = extended

    def get_coupled_variables(self, variables):

        tor_0 = variables["Leading-order electrolyte tortuosity"]
        c_e_0_av = variables["Leading-order x-averaged electrolyte concentration"]
        c_e = variables["Electrolyte concentration"]
        # i_e = variables["Electrolyte current density"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]
        T_0 = variables["Leading-order cell temperature"]

        param = self.param

        N_e_diffusion = -tor_0 * param.D_e(c_e_0_av, T_0) * pybamm.grad(c_e)
        # N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        # N_e_convection = c_e * v_box_0

        # N_e = N_e_diffusion + N_e_migration + N_e_convection

        if v_box_0.id == pybamm.Scalar(0).id:
            N_e = N_e_diffusion
        else:
            N_e = N_e_diffusion + v_box_0 * c_e

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
        else:
            source_terms_0 = self._get_source_terms_first_order(variables)

        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms_0 - c_e * deps_0_dt)
        }

    def _get_source_terms_leading_order(self, variables):
        return sum(
            pybamm.Concatenation(
                reaction["Negative"]["s"]
                * variables["Leading-order " + reaction["Negative"]["aj"].lower()],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                reaction["Positive"]["s"]
                * variables["Leading-order " + reaction["Positive"]["aj"].lower()],
            )
            / self.param.gamma_e
            for reaction in self.reactions.values()
        )

    def _get_source_terms_first_order(self, variables):
        return sum(
            pybamm.Concatenation(
                reaction["Negative"]["s"] * variables[reaction["Negative"]["aj"]],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                reaction["Positive"]["s"] * variables[reaction["Positive"]["aj"]],
            )
            / self.param.gamma_e
            for reaction in self.reactions.values()
        )
