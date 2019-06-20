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

    *Extends:* :class:`pybamm.BaseStefanMaxwellDiffusion`
    """

    def __init__(self, param, ocp=False):
        super().__init__(param, ocp)

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel which can be stated independent of
        variables stated in other submodels
        """
        c_e_av = pybamm.standard_variables.c_e_av
        c_e_n = pybamm.Broadcast(c_e_av, ["negative electrode"])
        c_e_s = pybamm.Broadcast(c_e_av, ["separator"])
        c_e_p = pybamm.Broadcast(c_e_av, ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

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

        c_e = variables["Average electrolyte concentration"]
        eps = variables["Porosity"]
        deps_n_dt = variables["Negative porosity change"]
        deps_p_dt = variables["Positive porosity change"]

        eps_n, eps_s, eps_p = [e.orphans[0] for e in eps.orphans]

        # TODO: ask tino about this bit
        # source_terms = sum(
        #     param.l_n * rxn["neg"]["s_plus"] * rxn["neg"]["aj"]
        #     + param.l_p * rxn["pos"]["s_plus"] * rxn["pos"]["aj"]
        #     for rxn in self.reactions.values()
        # )
        source_terms = 0

        self.rhs = {
            c_e: 1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (source_terms - c_e * (param.l_n * deps_n_dt + param.l_p * deps_p_dt))
        }

    def set_initial_conditions(self, variables):
        c_e = variables["Average electrolyte concentration"]
        self.initial_conditions = {c_e: self.param.c_e_init}
