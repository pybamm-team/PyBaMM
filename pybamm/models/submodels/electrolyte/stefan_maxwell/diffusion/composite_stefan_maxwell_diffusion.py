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

        eps_0 = variables["Leading-order porosity"]
        c_e_0 = variables["Leading-order average electrolyte concentration"]
        c_e = variables["Electrolyte concentration"]
        # i_e = variables["Electrolyte current density"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]
        T_0 = variables["Leading-order cell temperature"]

        param = self.param

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        N_e_diffusion = (
            -(eps_0 ** param.b)
            * pybamm.PrimaryBroadcast(param.D_e(c_e_0, T_0), whole_cell)
            * pybamm.grad(c_e)
        )
        # N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        # N_e_convection = c_e * v_box_0

        # N_e = N_e_diffusion + N_e_migration + N_e_convection

        if v_box_0.id == pybamm.Scalar(0).id:
            N_e = N_e_diffusion
        else:
            N_e = N_e_diffusion + pybamm.outer(v_box_0, c_e)

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
            neg_reactions = sum(
                reaction["Negative"]["s"]
                * variables["Leading-order " + reaction["Negative"]["aj"].lower()]
                for reaction in self.reactions.values()
            )
            pos_reactions = sum(
                reaction["Positive"]["s"]
                * variables["Leading-order " + reaction["Positive"]["aj"].lower()]
                for reaction in self.reactions.values()
            )
        else:
            neg_reactions = sum(
                reaction["Negative"]["s"] * variables[reaction["Negative"]["aj"]]
                for reaction in self.reactions.values()
            )
            pos_reactions = sum(
                reaction["Positive"]["s"] * variables[reaction["Positive"]["aj"]]
                for reaction in self.reactions.values()
            )
        sep_reactions = pybamm.FullBroadcast(0, "separator", "current collector")
        source_terms_0 = (
            pybamm.Concatenation(neg_reactions, sep_reactions, pos_reactions)
            / param.gamma_e
        )

        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms_0 - c_e * deps_0_dt)
        }
