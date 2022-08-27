#
# No convection
#
import pybamm
from .base_through_cell_convection import BaseThroughCellModel


class NoConvection(BaseThroughCellModel):
    """A submodel for case where there is no convection.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.convection.through_cell.BaseThroughCellModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        vel_scale = self.param.velocity_scale
        acc_scale = vel_scale / self.param.L_x

        variables = {}
        domains = [domain for domain in self.domains if domain != "Separator"]
        for domain in domains:
            v_box_k = pybamm.FullBroadcast(0, domain.lower(), "current collector")
            div_v_box_k = pybamm.FullBroadcast(0, domain.lower(), "current collector")
            div_v_box_k_av = pybamm.x_average(div_v_box_k)
            p_k = pybamm.FullBroadcast(0, domain.lower(), "current collector")

            variables.update(
                {
                    f"{domain} volume-averaged velocity": v_box_k,
                    f"{domain} volume-averaged velocity [m.s-1]": vel_scale * v_box_k,
                    f"{domain} volume-averaged acceleration": div_v_box_k,
                    f"{domain} volume-averaged acceleration [m.s-1]": acc_scale
                    * div_v_box_k,
                    f"X-averaged {domain.lower()} volume-averaged acceleration"
                    + "": div_v_box_k_av,
                    f"X-averaged {domain.lower()} volume-averaged acceleration "
                    + "[m.s-1]": acc_scale * div_v_box_k_av,
                    f"{domain} pressure": p_k,
                    f"X-averaged {domain.lower()} pressure": pybamm.x_average(p_k),
                }
            )

        return variables

    def get_coupled_variables(self, variables):

        # Simple formula for velocity in the separator
        v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")
        div_v_box_s = pybamm.FullBroadcast(0, "separator", "current collector")

        variables.update(
            self._get_standard_sep_velocity_variables(v_box_s, div_v_box_s)
        )
        variables.update(self._get_standard_whole_cell_velocity_variables(variables))
        variables.update(
            self._get_standard_whole_cell_acceleration_variables(variables)
        )
        variables.update(self._get_standard_whole_cell_pressure_variables(variables))

        return variables
