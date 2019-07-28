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

        return variables

    def get_coupled_variables(self, variables):
        param = self.param

        phi_s_p = variables["Positive electrode potential"]
        phi_s_cn = variables["Negative current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        # The voltage-current expression from the SPM(e)
        # note that phi_s_cn is equal pybamm.boundary_value(phi_s_n, "left")
        voltage_from_1D_models = pybamm.boundary_value(phi_s_p, "right") - phi_s_cn
        phi_s_cp = phi_s_cn + voltage_from_1D_models
        variables = self._get_standard_potential_variables(phi_s_cn, phi_s_cp)
        return variables
