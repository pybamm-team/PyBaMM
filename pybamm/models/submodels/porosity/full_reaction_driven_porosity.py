#
# Class for reaction driven porosity changes
#
import pybamm
from .base_porosity import BaseModel


class Full(BaseModel):
    """Full model for reaction-driven porosity changes

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        eps_n = pybamm.standard_variables.eps_n
        eps_s = pybamm.standard_variables.eps_s
        eps_p = pybamm.standard_variables.eps_p
        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)

        return variables

    def get_coupled_variables(self, variables):
        j_sei = variables["Negative electrode sei interfacial current density"]
        C_sei_eps = pybamm.sei_parameters.C_sei_eps

        deps_n_dt = C_sei_eps * j_sei
        deps_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )
        # for the moment is not needed but could be used for positive SEI
        deps_p_dt = pybamm.FullBroadcast(0, "positive electrode",
                                         "current collector")

        variables.update(
            self._get_standard_porosity_change_variables(
                deps_n_dt, deps_s_dt, deps_p_dt
            )
        )

        return variables

    def set_rhs(self, variables):

        eps = variables["Porosity"]
        deps_dt = variables["Porosity change"]

        self.rhs = {eps: deps_dt}

    def set_initial_conditions(self, variables):
        eps = variables["Porosity"]
        self.initial_conditions = {eps: self.param.epsilon_init}
