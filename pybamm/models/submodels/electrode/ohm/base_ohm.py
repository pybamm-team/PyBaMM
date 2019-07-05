#
# Base class for Ohm's law submodels
#
import pybamm
from ..base_electrode import BaseElectrode


class BaseModel(BaseElectrode):
    """A base class for electrode submodels that employ
    Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.electrode.BaseElectrode`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
