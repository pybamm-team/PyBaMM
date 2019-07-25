#
# Base class for electrolyte conductivity employing stefan-maxwell
#
from ...base_electrolyte_conductivity import BaseElectrolyteConductivity
import pybamm


class BaseModel(BaseElectrolyteConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte.BaseElectrolyteConductivity`
    """

    def __init__(self, param, domain=None, reactions=None):
        super().__init__(param, domain, reactions)

    def set_boundary_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        self.boundary_conditions = {
            phi_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }
