#
# Class for varying active material volume fraction
#
import pybamm

from .base_active_material import BaseModel


class VaryingFull(BaseModel):
    """Submodel for varying active material volume fraction

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`pybamm.active_material.BaseModel`
    """

    def get_fundamental_variables(self):
        eps_solid = pybamm.Variable(
            self.domain + " electrode active material volume fraction",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        variables = self._get_standard_active_material_variables(eps_solid)
        return variables

    def get_coupled_variables(self, variables):
        # This submodel only contains the structure to allow the active material volume
        # fraction to vary, it does not implement an actual model for LAM.
        # As a placeholder, we use deps_dt = 0 * j
        j = variables[self.domain + " electrode interfacial current density"]
        deps_solid_dt = 0 * j
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        eps_solid = variables[
            self.domain + " electrode active material volume fraction"
        ]
        deps_solid_dt = variables[
            self.domain + " electrode active material volume fraction change"
        ]
        self.rhs = {eps_solid: deps_solid_dt}

    def set_initial_conditions(self, variables):
        eps_solid = variables[
            self.domain + " electrode active material volume fraction"
        ]

        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            eps_solid_init = self.param.epsilon_s_n(x_n)
        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            eps_solid_init = self.param.epsilon_s_p(x_p)

        self.initial_conditions = {eps_solid: eps_solid_init}
