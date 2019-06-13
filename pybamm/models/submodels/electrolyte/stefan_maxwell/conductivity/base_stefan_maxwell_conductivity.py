#
# Base class for electrolyte conductivity employing stefan-maxwell
#
from ...base_electrolyte_conductivity import BaseElectrolyteConductivity


class BaseStefanMaxwellConductivity(BaseElectrolyteConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)
