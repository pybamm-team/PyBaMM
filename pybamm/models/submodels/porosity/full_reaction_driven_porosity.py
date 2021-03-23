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

    def __init__(self, param, options):
        super().__init__(param, options)

    def get_fundamental_variables(self):

        eps_n = pybamm.standard_variables.eps_n
        eps_s = pybamm.standard_variables.eps_s
        eps_p = pybamm.standard_variables.eps_p
        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)

        return variables

    def get_coupled_variables(self, variables):

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]
        deps_n_dt = -self.param.beta_surf_n * j_n

        if self.options["SEI porosity change"] == "true":

            j_sei_n = variables["Negative electrode SEI interfacial current density"]
            beta_sei_n = self.param.beta_sei_n
            deps_n_dt += beta_sei_n * j_sei_n

        if self.options["lithium plating porosity change"] == "true":

            j_plating = variables[
                "Negative electrode lithium plating interfacial current density"
            ]
            beta_plating = self.param.beta_plating

            deps_n_dt += beta_plating * j_plating

        deps_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )

        deps_p_dt = -self.param.beta_surf_p * j_p

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
