#
# Class for constant active material
#
import pybamm

from .base_active_material import BaseModel


class Constant(BaseModel):
    """Submodel for constant active material

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.loss_of_active_material.BaseModel`
    """

    def get_fundamental_variables(self):

        eps_am_n = pybamm.FullBroadcast(
            self.param.epsilon_s_n, "negative electrode", "current collector"
        )
        eps_am_s = pybamm.FullBroadcast(0, "separator", "current collector")
        eps_am_p = pybamm.FullBroadcast(
            self.param.epsilon_s_p, "positive electrode", "current collector"
        )

        deps_am_n_dt = pybamm.FullBroadcast(
            0, "negative electrode", "current collector"
        )
        deps_am_s_dt = pybamm.FullBroadcast(0, "separator", "current collector")
        deps_am_p_dt = pybamm.FullBroadcast(
            0, "positive electrode", "current collector"
        )

        variables = self._get_standard_activate_material_variables(
            eps_am_n, eps_am_s, eps_am_p, set_leading_order=True
        )
        variables.update(
            self._get_standard_active_material_change_variables(
                deps_am_n_dt, deps_am_s_dt, deps_am_p_dt, set_leading_order=True
            )
        )

        return variables
