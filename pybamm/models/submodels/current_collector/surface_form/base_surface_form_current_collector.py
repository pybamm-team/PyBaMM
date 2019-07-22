#
# Base class for current collectors with surface formulation
#
from ..base_current_collector import BaseModel


class BaseSurfaceForm(BaseModel):
    """A base submodel for Ohm's law plus conservation of current in the current
    collectors, using the surface formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)
