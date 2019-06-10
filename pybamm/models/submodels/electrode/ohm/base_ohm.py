#
# Base class for Ohm's law submodels
#
import pybamm


class BaseOhm(pybamm.BaseElectrode):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
