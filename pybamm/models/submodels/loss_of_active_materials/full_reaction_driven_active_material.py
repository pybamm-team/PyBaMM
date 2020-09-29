#
# Class for reaction driven active material volume fraction changes
#
import pybamm
from .base_active_material import BaseModel


class Full(BaseModel):
    """Full model for reaction-driven active material volume fraction changes

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.loss_of_active_materials.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        eps_am_n = pybamm.standard_variables.eps_am_n
        eps_am_s = pybamm.standard_variables.eps_am_s
        eps_am_p = pybamm.standard_variables.eps_am_p
        variables = self._get_standard_activate_material_variables(
            eps_am_n, eps_am_s, eps_am_p
        )

        return variables

    def get_coupled_variables(self, variables):
        if "Negative particle surface tangential stress" in variables:
            stress_t_surf_n = variables["Negative particle surface tangential stress"]
            stress_t_surf_n *= stress_t_surf_n > 0
        else:
            stress_t_surf_n = pybamm.FullBroadcast(
                0,
                "negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        if "Positive particle surface tangential stress" in variables:
            stress_t_surf_p = variables["Positive particle surface tangential stress"]
            stress_t_surf_p *= stress_t_surf_p > 0
        else:
            stress_t_surf_p = pybamm.FullBroadcast(
                0,
                "positive electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

        mp = pybamm.mechanical_parameters

        deps_am_n_dt = (
            -mp.beta_LAM_n
            * pybamm.Power(stress_t_surf_n / mp.stress_c_n, mp.m_LAM_n)
            / mp.t0_cr
        )
        deps_am_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )
        deps_am_p_dt = (
            -mp.beta_LAM_p
            * pybamm.Power(stress_t_surf_p / mp.stress_c_p, mp.m_LAM_p)
            / mp.t0_cr
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
        eps_am_n = pybamm.FullBroadcast(
            self.param.epsilon_s_n, "negative electrode", "current collector"
        )
        eps_am_s = pybamm.FullBroadcast(0, "separator", "current collector")
        eps_am_p = pybamm.FullBroadcast(
            self.param.epsilon_s_p, "positive electrode", "current collector"
        )
        eps_am_init = pybamm.Concatenation(eps_am_n, eps_am_s, eps_am_p)
        self.initial_conditions = {eps_am: eps_am_init}
