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
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        phi_s_cn = pybamm.PrimaryBroadcast(0, "current collector")
        variables = self._get_standard_negative_potential_variables(phi_s_cn)
        return variables

    def get_coupled_variables(self, variables):
        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.PrimaryBroadcast(
            variables["Total current density [A.m-2]"], "current collector"
        )

        variables = self._get_standard_current_variables(i_cc, i_boundary_cc)

        return variables
