#
# Class for varying active material volume fraction
#
import pybamm

from .base_active_material import BaseModel


class VaryingUniform(BaseModel):
    """Submodel for varying active material volume fraction, with variations uniform
    across each electrode

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
        domain = self.domain.lower() + " electrode"
        eps_solid_xav = pybamm.Variable(
            "X-averaged " + domain + " active material volume fraction",
            domain="current collector",
        )
        eps_solid = pybamm.PrimaryBroadcast(eps_solid_xav, domain)
        variables = self._get_standard_active_material_variables(eps_solid)
        return variables

    def get_coupled_variables(self, variables):
        domain = self.domain.lower() + " electrode"
        # This is loss of active material model by mechanical effects
        if (
            "X-averaged " + self.domain.lower() + " particle surface tangential stress"
            in variables
        ):
            stress_t_surf = variables[
                "X-averaged "
                + self.domain.lower()
                + " particle surface tangential stress"
            ]
            stress_t_surf *= stress_t_surf > 0
            stress_r_surf = variables[
                "X-averaged " + self.domain.lower() + " particle surface radial stress"
            ]
            stress_r_surf *= stress_r_surf > 0
        else:
            stress_t_surf = pybamm.Scalar(0)
            stress_r_surf = stress_t_surf
        if self.domain == "Negative":
            beta_LAM = self.param.beta_LAM_n
            stress_critical = self.param.stress_critical_n
            m_LAM = self.param.m_LAM_n
        else:
            beta_LAM = self.param.beta_LAM_p
            stress_critical = self.param.stress_critical_p
            m_LAM = self.param.m_LAM_p
        stress_h_surf = (stress_r_surf + 2 * stress_t_surf) / 3
        # assuming the minimum hydrostatic stress is zero
        stress_h_surf_min = stress_h_surf * 0
        deps_solid_dt_xav = (
            -beta_LAM
            * pybamm.Power(
                (stress_h_surf - stress_h_surf_min) / stress_critical,
                m_LAM,
            )
            / self.param.t0_cr
        )

        deps_solid_dt = pybamm.PrimaryBroadcast(deps_solid_dt_xav, domain)
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower() + " electrode"
        eps_solid_xav = variables[
            "X-averaged " + domain + " active material volume fraction"
        ]
        deps_solid_dt_xav = variables[
            "X-averaged " + domain + " active material volume fraction change"
        ]
        self.rhs = {eps_solid_xav: deps_solid_dt_xav}

    def set_initial_conditions(self, variables):
        eps_solid_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode active material volume fraction"
        ]

        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            eps_solid_init = self.param.epsilon_s_n(x_n)
        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            eps_solid_init = self.param.epsilon_s_p(x_p)

        self.initial_conditions = {eps_solid_xav: pybamm.x_average(eps_solid_init)}
