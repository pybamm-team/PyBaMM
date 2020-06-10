#
# Base class for Li plating models.
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for particle cracking models.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms
    **Extends:** :class:`pybamm.BaseSubModel`
    """

        return variables