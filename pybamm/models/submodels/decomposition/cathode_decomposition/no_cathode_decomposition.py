#
# Class for cathode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class NoCathodeDecomposition(pybamm.BaseSubModel):
    """Base class for cathode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, reactions=None):
        super().__init__(param)

    def get_fundamental_variables(self):
        default = pybamm.Scalar(0)
        variables = {
            "Degree of conversion of cathode decomposition": default,
            "Cathode decomposition reaction rate [s-1]": default,
            "Cathode decomposition reaction rate": default,
            "Cathode decomposition heating": default,
            "Cathode decomposition heating [W.m-3]": default,

            "Core section cathode decomposition reaction rate [s-1]": default,
            "Core section cathode decomposition reaction rate": default,
            "Core section cathode decomposition heating": default,
            "Core section cathode decomposition heating [W.m-3]": default,

            "Middle section cathode decomposition reaction rate [s-1]": default,
            "Middle section cathode decomposition reaction rate": default,
            "Middle section cathode decomposition heating": default,
            "Middle section cathode decomposition heating [W.m-3]": default,

            "Outer section cathode decomposition reaction rate [s-1]": default,
            "Outer section cathode decomposition reaction rate": default,
            "Outer section cathode decomposition heating": default,
            "Outer section cathode decomposition heating [W.m-3]": default,

        }
        return variables