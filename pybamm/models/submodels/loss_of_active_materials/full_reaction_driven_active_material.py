#
# Class for reaction driven active material volume fraction changes
#
import pybamm
from .base_active_material import BaseLAM


class Full(BaseLAM):
    """Full model for reaction-driven active material volume fraction changes
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
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

        eps_am_n = pybamm.standard_variables.eps_am_n
        eps_am_s = pybamm.standard_variables.eps_am_s
        eps_am_p = pybamm.standard_variables.eps_am_p
        variables = self._get_standard_active_material_variables(
            eps_am_n, eps_am_s, eps_am_p
        )

        return variables

    def get_coupled_variables(self, variables):
        if "Negative particle surface tangential stress" in variables:
            stress_t_surf_n = variables["Negative particle surface tangential stress"]
            stress_t_surf_n *= stress_t_surf_n > 0
            stress_r_surf_n = variables["Negative particle surface radial stress"]
            stress_r_surf_n *= stress_r_surf_n > 0
        else:
            stress_t_surf_n = pybamm.FullBroadcast(
                0,
                "negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            stress_r_surf_n = stress_t_surf_n
        if "Positive particle surface tangential stress" in variables:
            stress_t_surf_p = variables["Positive particle surface tangential stress"]
            stress_t_surf_p *= stress_t_surf_p > 0
            stress_r_surf_p = variables["Positive particle surface radial stress"]
            stress_r_surf_p *= stress_r_surf_p > 0
        else:
            stress_t_surf_p = pybamm.FullBroadcast(
                0,
                "positive electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            stress_r_surf_p *= stress_r_surf_p > 0

        stress_h_surf_p = (stress_r_surf_p + 2 * stress_t_surf_p) / 3
        stress_h_surf_n = (stress_r_surf_n + 2 * stress_t_surf_n) / 3
        # assuming the minimum hydrostatic stress is zero
        stress_h_surf_p_min = stress_h_surf_p * 0
        stress_h_surf_n_min = stress_h_surf_n * 0
        deps_am_n_dt = (
            -self.param.beta_LAM_n
            * pybamm.Power(
                (stress_h_surf_n - stress_h_surf_n_min) / self.param.stress_critical_n,
                self.param.m_LAM_n,
            )
            / self.param.t0_cr
        )
        deps_am_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )
        deps_am_p_dt = (
            -self.param.beta_LAM_p
            * pybamm.Power(
                (stress_h_surf_p - stress_h_surf_p_min) / self.param.stress_critical_p,
                self.param.m_LAM_p,
            )
            / self.param.t0_cr
        )

        variables.update(
            self._get_standard_active_material_change_variables(
                deps_am_n_dt, deps_am_s_dt, deps_am_p_dt
            )
        )

        return variables

    def set_rhs(self, variables):

        eps_am = variables["Active material volume fraction"]
        deps_am_dt = variables["Active material volume fraction change"]

        self.rhs = {eps_am: deps_am_dt}

    def set_initial_conditions(self, variables):
        eps_am = variables["Active material volume fraction"]
        self.initial_conditions = {eps_am: self.param.eps_am_init}
