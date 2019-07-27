#
# Class for the composite electrolyte potential employing stefan-maxwell
#
import pybamm
from .base_higher_order_stefan_maxwell_conductivity import BaseHigherOrder


class Composite(BaseHigherOrder):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Composite refers to a composite
    leading and first-order expression from the asymptotic reduction)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds

    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.BaseHigerOrder`
    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)

    def _higher_order_macinnes_function(self, x):
        "Use log for composite higher order terms"
        return pybamm.log(x)

    def unpack(self, variables):
        "Unpack variables and return average values"
        c_e_av = variables["X-averaged electrolyte concentration"]
        return c_e_av
