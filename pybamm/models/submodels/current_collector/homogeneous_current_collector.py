#
# Class for uniform current collectors
#
import pybamm
from .base_current_collector import BaseModel


class Uniform(BaseModel):
    """A submodel for uniform potential in the current collectors which
    is valid in the limit of fast conductivity in the current collectors.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.PrimaryBroadcast(
            self.param.current_with_time, "current collector"
        )
        phi_s_cn = pybamm.PrimaryBroadcast(0, "current collector")

        variables = self._get_standard_negative_potential_variables(phi_s_cn)
        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        # Hack to get the leading-order current collector current density
        # Note that this should be different from the actual (composite) current
        # collector current density for 2+1D models, but not sure how to implement this
        # using current structure of lithium-ion models
        variables["Leading-order current collector current density"] = variables[
            "Current collector current density"
        ]
        return variables

    def get_coupled_variables(self, variables):
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_p = variables["Positive electrode potential"]
        phi_s_cp = pybamm.boundary_value(phi_s_p, "right")
        variables = {
            "Positive current collector potential": phi_s_cp,
        }
        variables.update(self._get_standard_potential_variables(phi_s_cn, phi_s_cp))
        return variables
