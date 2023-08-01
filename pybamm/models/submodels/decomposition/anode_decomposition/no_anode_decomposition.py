#
# Class for when there is no anode decomposition
#
import pybamm
from scipy import constants


class NoAnodeDecomposition(pybamm.BaseSubModel):
    """Base class for no graphite anode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        default = pybamm.Scalar(0)
        variables = {
            "Relative SEI thickness": default,
            "Anode decomposition reaction rate [s-1]": default,
            "Anode decomposition reaction rate": default,
            "Anode decomposition heating": default,
            "Anode decomposition heating [W.m-3]": default,
            
            "Core section anode decomposition reaction rate [s-1]": default,
            "Core section anode decomposition reaction rate": default,
            "Core section anode decomposition heating": default,
            "Core section anode decomposition heating [W.m-3]": default,

            "Middle section anode decomposition reaction rate [s-1]": default,
            "Middle section anode decomposition reaction rate": default,
            "Middle section anode decomposition heating": default,
            "Middle section anode decomposition heating [W.m-3]": default,

            "Outer section anode decomposition reaction rate [s-1]": default,
            "Outer section anode decomposition reaction rate": default,
            "Outer section anode decomposition heating": default,
            "Outer section anode decomposition heating [W.m-3]": default,
        }

        return variables
