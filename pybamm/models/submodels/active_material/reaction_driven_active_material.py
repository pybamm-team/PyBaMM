#
# Class for varying active material volume fraction, driven by reactions
#
import pybamm

from .base_active_material import BaseModel


class ReactionDriven(BaseModel):
    """Submodel for varying active material volume fraction, driven by reactions, from
    [1]_

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)

    **Extends:** :class:`pybamm.active_material.BaseModel`

    References
    ----------
    .. [1] Reniers, J. M., Mulder, G., & Howey, D. A. (2019). Review and performance
           comparison of mechanical-chemical degradation models for lithium-ion
           batteries. Journal of The Electrochemical Society, 166(14), A3189.
    """

    def __init__(self, param, domain, options, x_average):
        super().__init__(param, domain, options=options)
        pybamm.citations.register("Reniers2019")
        self.x_average = x_average

    def get_fundamental_variables(self):
        domain = self.domain.lower() + " electrode"
        if self.x_average is True:
            eps_solid_xav = pybamm.Variable(
                "X-averaged " + domain + " active material volume fraction",
                domain="current collector",
            )
            eps_solid = pybamm.PrimaryBroadcast(eps_solid_xav, domain)
        else:
            eps_solid = pybamm.Variable(
                self.domain + " electrode active material volume fraction",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
            )
        variables = self._get_standard_active_material_variables(eps_solid)
        return variables

    def get_coupled_variables(self, variables):
        if self.x_average is True:
            a = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode surface area to volume ratio"
            ]
        else:
            a = variables[self.domain + " electrode surface area to volume ratio"]

        if self.domain == "Negative":
            beta_LAM_sei = self.param.beta_LAM_sei_n
            if self.x_average is True:
                j_sei = variables["X-averaged SEI interfacial current density"]
            else:
                j_sei = variables["SEI interfacial current density"]
        else:
            beta_LAM_sei = self.param.beta_LAM_sei_p
            j_sei = 0

        deps_solid_dt = beta_LAM_sei * a * j_sei
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        if self.x_average is True:
            eps_solid = variables[
                "X-averaged " + domain + " active material volume fraction"
            ]
            deps_solid_dt = variables[
                "X-averaged " + domain + " active material volume fraction change"
            ]
        else:
            eps_solid = variables[
                self.domain + " electrode active material volume fraction"
            ]
            deps_solid_dt = variables[
                self.domain + " electrode active material volume fraction change"
            ]

        self.rhs = {eps_solid: deps_solid_dt}

    def set_initial_conditions(self, variables):

        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            eps_solid_init = self.param.epsilon_s_n(x_n)
        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            eps_solid_init = self.param.epsilon_s_p(x_p)

        if self.x_average is True:
            eps_solid_xav = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode active material volume fraction"
            ]
            self.initial_conditions = {eps_solid_xav: pybamm.x_average(eps_solid_init)}
        else:
            eps_solid = variables[
                self.domain + " electrode active material volume fraction"
            ]
            self.initial_conditions = {eps_solid: eps_solid_init}
