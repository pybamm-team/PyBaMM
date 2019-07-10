#
# Class for electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_diffusion import BaseModel


class Composite(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_fundamental_variables(self):
        c_e = pybamm.standard_variables.c_e

        return self._get_standard_concentration_variables(c_e)

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

        import ipdb

        ipdb.set_trace()
        self.rhs = {
            c_e: (1 / eps_0)
            * (-pybamm.div(N_e) / param.C_e + source_terms_0 - c_e * deps_0_dt)
        }

    def set_boundary_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.initial_conditions = {c_e: self.param.c_e_init}
