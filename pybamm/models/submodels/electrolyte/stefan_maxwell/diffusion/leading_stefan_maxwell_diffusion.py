#
# Class for leading-order electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_diffusion import BaseModel


class LeadingOrderModel(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Leading refers to leading order
    of asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.diffusion.BaseModel`
    """

    def __init__(self, param, reactions, ocp=False):
        super().__init__(param, ocp)
        self.reactions = reactions

    def get_fundamental_variables(self):
        c_e_av = pybamm.standard_variables.c_e_av
        c_e_n = pybamm.Broadcast(c_e_av, ["negative electrode"])
        c_e_s = pybamm.Broadcast(c_e_av, ["separator"])
        c_e_p = pybamm.Broadcast(c_e_av, ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        c_e_n.evaluate_for_shape()
        variables = self._get_standard_concentration_variables(c_e, c_e_av)

        if self.ocp is True:
            variables.update(self._get_standard_ocp_variables(c_e))

        return variables

    def get_coupled_variables(self, variables):

        N_e = pybamm.Broadcast(
            0, ["negative electrode", "separator", "positive electrode"]
        )

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        c_e_av = variables["Average electrolyte concentration"]

        eps_n_av = variables["Average negative electrode porosity"]
        eps_s_av = variables["Average separator porosity"]
        eps_p_av = variables["Average positive electrode porosity"]

        deps_n_dt_av = variables["Average negative electrode porosity change"]
        deps_p_dt_av = variables["Average positive electrode porosity change"]

        # TODO: ask tino about this bit
        # the "j" component of reactions is now just a string referring to the variable
        # so that the expression doesn't need to be set twice
        source_terms = sum(
            param.l_n * rxn["neg"]["s_plus"] * variables[rxn["neg"]["j"]]
            + param.l_p * rxn["pos"]["s_plus"] * variables[rxn["pos"]["j"]]
            for rxn in self.reactions.values()
        )

        # source_terms = sum(
        #     param.l_n * rxn["neg"]["s_plus"] * rxn["neg"]["aj"]
        #     + param.l_p * rxn["pos"]["s_plus"] * rxn["pos"]["aj"]
        #     for rxn in self.reactions.values()
        # )

        self.rhs = {
            c_e_av: 1
            / (param.l_n * eps_n_av + param.l_s * eps_s_av + param.l_p * eps_p_av)
            * (
                source_terms
                - c_e_av * (param.l_n * deps_n_dt_av + param.l_p * deps_p_dt_av)
            )
        }

    def set_initial_conditions(self, variables):
        c_e = variables["Average electrolyte concentration"]
        self.initial_conditions = {c_e: self.param.c_e_init}
