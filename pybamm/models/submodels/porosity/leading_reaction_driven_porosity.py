#
# Class for leading-order reaction driven porosity changes
#
import pybamm
from .base_porosity import BaseModel


class LeadingOrder(BaseModel):
    """Class for reaction-driven porosity changes

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        eps = pybamm.standard_variables.eps_piecewise_constant
        variables = self._get_standard_porosity_variables(eps)
        return variables

    def get_coupled_variables(self, variables):

        j_n_av = variables["Average negative electrode interfacial current density"]
        j_s_av = pybamm.Scalar(0)
        j_p_av = variables["Average positive electrode interfacial current density"]

        deps_dt_n = pybamm.Broadcast(
            -self.param.beta_surf * j_n_av, ["negative electrode"]
        )
        deps_dt_s = pybamm.Broadcast(-self.param.beta_surf * j_s_av, ["separator"])
        deps_dt_p = pybamm.Broadcast(
            -self.param.beta_surf * j_p_av, ["positive electrode"]
        )

        deps_dt = pybamm.Concatenation(deps_dt_n, deps_dt_s, deps_dt_p)

        variables.update(self._get_standard_porosity_change_variables(deps_dt))

        return variables

    def set_rhs(self, variables):
        for domain in ["negative electrode", "separator", "positive electrode"]:
            eps_av = variables["Average " + domain + " porosity"]
            deps_dt_av = variables["Average " + domain + " porosity change"]
            self.rhs = {eps_av: deps_dt_av}

    def set_initial_conditions(self, variables):
        for domain in ["negative electrode", "separator", "positive electrode"]:
            eps_av = variables["Average " + domain + " porosity"]
            self.initial_conditions = {eps_av: self.param.eps_init}
