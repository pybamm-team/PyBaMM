#
# Class for varying active material volume fraction, driven by stress
#
import pybamm

from .base_active_material import BaseModel


class LossActiveMaterial(BaseModel):
    """Submodel for varying active material volume fraction from [1]_ and [2]_.

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
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2019). Electrochemical
           Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch
           Cells. Journal of The Electrochemical Society, 167(1), 013512.
    .. [2] Reniers, J. M., Mulder, G., & Howey, D. A. (2019). Review and performance
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
        deps_solid_dt = 0
        lam_option = getattr(self.options, self.domain.lower())[
            "loss of active material"
        ]
        if "stress" in lam_option:
            # obtain the rate of loss of active materials (LAM) by stress
            # This is loss of active material model by mechanical effects
            if self.x_average is True:
                stress_t_surf = variables[
                    "X-averaged "
                    + self.domain.lower()
                    + " particle surface tangential stress"
                ]
                stress_r_surf = variables[
                    "X-averaged "
                    + self.domain.lower()
                    + " particle surface radial stress"
                ]
            else:
                stress_t_surf = variables[
                    self.domain + " particle surface tangential stress"
                ]
                stress_r_surf = variables[
                    self.domain + " particle surface radial stress"
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
                * ((stress_h_surf - stress_h_surf_min) / stress_critical) ** m_LAM
            )
            deps_solid_dt += j_stress_LAM

        if "reaction" in lam_option:
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
                # No SEI in the positive electrode so no reaction-driven LAM
                # until other reactions are implemented
                beta_LAM_sei = self.param.beta_LAM_sei_p
                j_sei = 0

            j_stress_reaction = beta_LAM_sei * a * j_sei
            deps_solid_dt += j_stress_reaction
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        Domain = self.domain + " electrode"
        if self.x_average is True:
            eps_solid = variables[
                "X-averaged " + Domain.lower() + " active material volume fraction"
            ]
            deps_solid_dt = variables[
                "X-averaged "
                + Domain.lower()
                + " active material volume fraction change"
            ]
        else:
            eps_solid = variables[Domain + " active material volume fraction"]
            deps_solid_dt = variables[
                Domain + " active material volume fraction change"
            ]

        self.rhs = {eps_solid: deps_solid_dt}

    def set_initial_conditions(self, variables):

        if self.domain == "Negative":
            eps_solid_init = self.param.epsilon_s_n
        elif self.domain == "Positive":
            eps_solid_init = self.param.epsilon_s_p

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
