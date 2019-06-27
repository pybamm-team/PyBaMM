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

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):

        i_cc = pybamm.Scalar(0)

        i_boundary_cc = self.param.current_with_time

        variables = self._get_standard_current_variables(i_cc, i_boundary_cc)

        return variables

