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

    def build(self):
        phi_s_cn = pybamm.PrimaryBroadcast(0, "current collector")
        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.PrimaryBroadcast(
            self.coupled_variable("Total current density [A.m-2]"), "current collector"
        )

        self.variables.update(self._get_standard_negative_potential_variables(phi_s_cn))
        self.variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))
