#
# Class for when there is no anode decomposition
#
import pybamm
from scipy import constants


class NoSeiDecomposition(pybamm.BaseSubModel):
    """Base class for no graphite anode decomposition in Li-ion batteries.

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
        # variables = {
        #     "Fraction of Li in SEI": pybamm.Scalar(0),
        #     "SEI decomposition reaction rate [s-1]": pybamm.Scalar(0),
        #     "SEI decomposition reaction rate": pybamm.Scalar(0),
        #     "SEI decomposition heating": pybamm.Scalar(0),
        #     "SEI decomposition heating [W.m-3]": pybamm.Scalar(0),
        # }
        variables = {
            "Fraction of Li in SEI": pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
            "SEI decomposition reaction rate [s-1]": pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
            "SEI decomposition reaction rate": pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
            "SEI decomposition heating": pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
            "SEI decomposition heating [W.m-3]": pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
        }

        return variables
