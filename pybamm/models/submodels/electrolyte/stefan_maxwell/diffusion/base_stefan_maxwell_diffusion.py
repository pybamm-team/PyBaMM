#
# Base class for electrolyte diffusion employing stefan-maxwell
#
from ...base_electrolyte_diffusion import BaseElectrolyteDiffusion


class BaseModel(BaseElectrolyteDiffusion):
    """Base class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, ocp=False):
        super().__init__(param, ocp)
