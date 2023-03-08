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
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        variables = {}
        for domain in self.options.whole_cell_domains:
            if domain != "separator":
                v_box_k = pybamm.FullBroadcast(0, domain, "current collector")
                div_v_box_k = pybamm.FullBroadcast(0, domain, "current collector")
                p_k = pybamm.FullBroadcast(0, domain, "current collector")

                variables.update(
                    self._get_standard_convection_variables(
                        domain, v_box_k, div_v_box_k, p_k
                    )
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
