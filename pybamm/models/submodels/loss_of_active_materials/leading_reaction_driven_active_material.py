#
# Class for leading-order reaction driven active material volume fraction changes
#
import pybamm
from .base_active_material import BaseLAM


class LeadingOrder(BaseLAM):
    """Leading-order model for reaction-driven active material volume fraction changes
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    TODO: add this to the leading order battery model
    **Extends:** :class:`pybamm.loss_of_active_materials.BaseModel`

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2019). Electrochemical
           Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch
           Cells. Journal of The Electrochemical Society, 167(1), 013512.
    .. [2] Reniers, J. M., Mulder, G., & Howey, D. A. (2019). Review and performance
           comparison of mechanical-chemical degradation models for lithium-ion
           batteries. Journal of The Electrochemical Society, 166(14), A3189.
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        eps_am_n_pc = pybamm.standard_variables.eps_am_n_pc
        eps_am_s_pc = pybamm.standard_variables.eps_am_s_pc
        eps_am_p_pc = pybamm.standard_variables.eps_am_p_pc

        eps_am_n = pybamm.PrimaryBroadcast(eps_am_n_pc, "negative electrode")
        eps_am_s = pybamm.PrimaryBroadcast(eps_am_s_pc, "separator")
        eps_am_p = pybamm.PrimaryBroadcast(eps_am_p_pc, "positive electrode")

        variables = self._get_standard_active_material_variables(
            eps_am_n, eps_am_s, eps_am_p
        )
        return variables

    def get_coupled_variables(self, variables):
        if "X-averaged negative particle surface tangential stress" in variables:
            stress_t_surf_n = variables[
                "X-averaged negative particle surface tangential stress"
            ]
            stress_t_surf_n *= stress_t_surf_n > 0
            stress_r_surf_n = variables[
                "X-averaged negative particle surface radial stress"
            ]
            stress_r_surf_n *= stress_r_surf_n > 0
        else:
            stress_t_surf_n = pybamm.FullBroadcast(
                0,
                "negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            stress_r_surf_n = stress_t_surf_n
        if "X-averaged positive particle surface tangential stress" in variables:
            stress_t_surf_p = variables[
                "X-averaged positive particle surface tangential stress"
            ]
            stress_t_surf_p *= stress_t_surf_p > 0
            stress_r_surf_p = variables[
                "X-averaged positive particle surface radial stress"
            ]
            stress_r_surf_p *= stress_r_surf_p > 0
        else:
            stress_t_surf_p = pybamm.FullBroadcast(
                0,
                "positive electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            stress_r_surf_p *= stress_t_surf_p
        stress_h_surf_p = (stress_r_surf_p + 2 * stress_t_surf_p) / 3
        stress_h_surf_n = (stress_r_surf_n + 2 * stress_t_surf_n) / 3
        # assuming the minimum hydrostatic stress is zero
        stress_h_surf_p_min = stress_h_surf_p * 0
        stress_h_surf_n_min = stress_h_surf_n * 0
        deps_am_n_dt = pybamm.PrimaryBroadcast(
            -self.param.beta_LAM_n
            * pybamm.Power(
                (stress_h_surf_n - stress_h_surf_n_min) / self.param.stress_critical_n,
                self.param.m_LAM_n,
            )
            / self.param.t0_cr,
            ["negative electrode"],
        )
        deps_am_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )
        deps_am_p_dt = pybamm.PrimaryBroadcast(
            -self.param.beta_LAM_p
            * pybamm.Power(
                (stress_h_surf_p - stress_h_surf_p_min) / self.param.stress_critical_p,
                self.param.m_LAM_p,
            )
            / self.param.t0_cr,
            ["positive electrode"],
        )

        variables.update(
            self._get_standard_active_material_change_variables(
                deps_am_n_dt, deps_am_s_dt, deps_am_p_dt
            )
        )

        return variables

    def set_rhs(self, variables):
        self.rhs = {}
        for domain in ["negative electrode", "separator", "positive electrode"]:
            eps_am_av = variables[
                "X-averaged " + domain + " active material volume fraction"
            ]
            deps_am_dt_av = variables[
                "X-averaged " + domain + " active material volume fraction change"
            ]
            self.rhs.update({eps_am_av: deps_am_dt_av})

    def set_initial_conditions(self, variables):

        eps_am_n_av = variables[
            "X-averaged negative electrode active material volume fraction"
        ]
        eps_am_s_av = variables["X-averaged separator active material volume fraction"]
        eps_am_p_av = variables[
            "X-averaged positive electrode active material volume fraction"
        ]

        self.initial_conditions = {eps_am_n_av: self.param.epsilon_s_n}
        self.initial_conditions.update({eps_am_s_av: 0})
        self.initial_conditions.update({eps_am_p_av: self.param.epsilon_s_p})
