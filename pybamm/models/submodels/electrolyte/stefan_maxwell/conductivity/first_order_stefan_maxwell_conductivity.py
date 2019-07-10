#
# Class for the composite electrolyte potential employing stefan-maxwell
#
import pybamm
from .base_higher_order_stefan_maxwell_conductivity import BaseHigherOrder


class FirstOrder(BaseHigherOrder):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (First order refers to a first-order
    expression from the asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.BaseHigerOrder`
    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)

    def _higher_order_macinnes_function(self, x):
        "Linear higher order terms"
        return x
