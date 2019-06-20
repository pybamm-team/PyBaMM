#
# Class for reaction driven porosity changes
#
import pybamm
from .base_porosity import BaseModel


class ReactionDriven(BaseModel):
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

        eps = pybamm.standard_variables.eps
        variables = self._get_standard_porosity_variables(eps)
        return variables

    def get_coupled_variables(self, variables):

        j = variables["Interfacial current density"]

        deps_dt = -self.param.beta_surf * j

        variables.update(self._get_standard_porosity_change_variables(deps_dt))

        return variables

    def set_rhs(self, variables):

        eps = variables["Porosity"]
        deps_dt = variables["Porosity change"]

        self.rhs = {eps: deps_dt}

    def set_initial_conditions(self, variables):
        eps = variables["Porosity"]
        self.initial_conditions = {eps: self.param.eps_init}
