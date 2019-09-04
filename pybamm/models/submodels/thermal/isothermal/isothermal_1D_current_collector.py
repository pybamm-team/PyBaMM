#
# Class for isothermal case which accounts for 1D current collectors
#
import pybamm

from .base_isothermal import BaseModel


class CurrentCollector1D(BaseModel):
    """Class for isothermal submodel with a 1D current collector.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)
