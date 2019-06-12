#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm


class FullSurfaceFormStefanMaxwellConductivity(
    pybamm.BaseSurfaceFormStefanMaxwellConductivity
):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseFullSurfaceFormStefanMaxwellConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def set_algebraic(self, variables):
        delta_phi = variables[self._domain + " electrode surface potential difference"]
        i_e = variables[self._domain + " electrolyte current density"]
        j = variables[self._domain + " electrode interfacial current density"]

        self.rhs[delta_phi] = pybamm.div(i_e) - j

