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

        i_cc = pybamm.SecondaryBroadcast(pybamm.Scalar(0), "current collector")

        i_boundary_cc = pybamm.PrimaryBroadcast(
            self.param.current_with_time, "current collector"
        )

        variables = self._get_standard_current_variables(i_cc, i_boundary_cc)

        return variables
