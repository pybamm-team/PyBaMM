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

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2019). Electrochemical
           Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch
           Cells. Journal of The Electrochemical Society, 167(1), 013512.
    .. [2] Reniers, J. M., Mulder, G., & Howey, D. A. (2019). Review and performance
           comparison of mechanical-chemical degradation models for lithium-ion
           batteries. Journal of The Electrochemical Society, 166(14), A3189.
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
        # obtain the rate of loss of activa materials (LAM) by stress
        j_stress_LAM = self._stress_driven_LAM_uniform(variables)
        # sum all LAM fluxes
        deps_solid_dt_xav = j_stress_LAM

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

    def _stress_driven_LAM_uniform(self, variables):
        # This is loss of active material model by mechanical effects
        stress_t_surf = variables[
            "X-averaged " + self.domain.lower() + " particle surface tangential stress"
        ]
        stress_r_surf = variables[
            "X-averaged " + self.domain.lower() + " particle surface radial stress"
        ]
        if self.domain == "Negative":
            beta_LAM = self.param.beta_LAM_n
            stress_critical = self.param.stress_critical_n
            m_LAM = self.param.m_LAM_n
        else:
            beta_LAM = self.param.beta_LAM_p
            stress_critical = self.param.stress_critical_p
            m_LAM = self.param.m_LAM_p

        stress_h_surf = (stress_r_surf + 2 * stress_t_surf) / 3
        # compressive stress make no contribution
        stress_h_surf *= stress_h_surf > 0
        # assuming the minimum hydrostatic stress is zero for full cycles
        stress_h_surf_min = stress_h_surf * 0
        j_stress_LAM = (
            -beta_LAM
            * pybamm.Power(
                (stress_h_surf - stress_h_surf_min) / stress_critical,
                m_LAM,
            )
            / self.param.t0_cr
        )
        return j_stress_LAM
