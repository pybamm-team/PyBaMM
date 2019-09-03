#
# Class for isothermal case
#
from .base_isothermal import BaseModel


class NoCurrentCollector(BaseModel):
    """Class for isothermal submodels with no current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _current_collector_heating(self, variables):
        """Returns zeros for current collector heat source terms"""
        Q_s_cn = 0
        Q_s_cp = 0
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z
            In this case this is just equal to the input variable"""
        return var
