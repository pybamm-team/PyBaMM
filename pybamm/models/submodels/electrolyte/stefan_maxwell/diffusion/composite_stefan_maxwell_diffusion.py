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

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.Full`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_coupled_variables(self, variables):

        eps_0 = variables["Leading-order porosity"]
        c_e_0 = variables["Leading-order average electrolyte concentration"]
        c_e = variables["Electrolyte concentration"]
        # i_e = variables["Electrolyte current density"]
        v_box_0 = variables["Leading-order volume-averaged velocity"]

        param = self.param

        N_e_diffusion = -(eps_0 ** param.b) * param.D_e(c_e_0) * pybamm.grad(c_e)
        # N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        # N_e_convection = c_e * v_box_0

        # N_e = N_e_diffusion + N_e_migration + N_e_convection

        N_e = N_e_diffusion + c_e * v_box_0

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps_0 = variables["Leading-order porosity"]
        deps_0_dt = variables["Leading-order porosity change"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        source_terms_0 = sum(
            pybamm.Concatenation(
                reaction["Negative"]["s"]
                * variables["Leading-order " + reaction["Negative"]["aj"].lower()],
                pybamm.Broadcast(0, "separator"),
                reaction["Positive"]["s"]
                * variables["Leading-order " + reaction["Positive"]["aj"].lower()],
            )
            / param.gamma_e
            for reaction in self.reactions.values()
        )

        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms_0 - c_e * deps_0_dt)
        }
