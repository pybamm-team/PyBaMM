#
# Base class for electrolyte conductivity employing stefan-maxwell
#
import pybamm


class BaseStefanMaxwellConductivity(pybamm.BaseElectrolyteConductivity):
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
