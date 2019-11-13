#
# Class for electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_diffusion import BaseModel


class Full(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions)

    def get_fundamental_variables(self):
        c_e = pybamm.standard_variables.c_e

        return self._get_standard_concentration_variables(c_e)

    def get_coupled_variables(self, variables):

        tor = variables["Electrolyte tortuosity"]
        c_e = variables["Electrolyte concentration"]
        # i_e = variables["Electrolyte current density"]
        v_box = variables["Volume-averaged velocity"]
        T = variables["Cell temperature"]

        param = self.param

        N_e_diffusion = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        # N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        # N_e_convection = c_e * v_box

        # N_e = N_e_diffusion + N_e_migration + N_e_convection

        N_e = N_e_diffusion + c_e * v_box

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps = variables["Porosity"]
        deps_dt = variables["Porosity change"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        # i_e = variables["Electrolyte current density"]

        # source_term = ((param.s - param.t_plus) / param.gamma_e) * pybamm.div(i_e)
        # source_term = pybamm.div(i_e) / param.gamma_e  # lithium-ion
        source_terms = sum(
            pybamm.Concatenation(
                reaction["Negative"]["s"] * variables[reaction["Negative"]["aj"]],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                reaction["Positive"]["s"] * variables[reaction["Positive"]["aj"]],
            )
            / param.gamma_e
            for reaction in self.reactions.values()
        )

        self.rhs = {
            c_e: (1 / eps)
            * (-pybamm.div(N_e) / param.C_e + source_terms - c_e * deps_dt)
        }

    def set_initial_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.initial_conditions = {c_e: self.param.c_e_init}
