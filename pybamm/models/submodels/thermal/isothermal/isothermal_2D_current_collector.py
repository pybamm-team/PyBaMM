#
# Class for isothermal case which accounts for current collectors
#
import pybamm

from .base_isothermal import BaseModel


class CurrentCollector2D(BaseModel):
    """Class for isothermal submodel with a 2D current collector"""

    def __init__(self, param):
        super().__init__(param)

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z"""
        return pybamm.yz_average(var)
