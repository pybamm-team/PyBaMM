#
# Class for leading-order surface form electrolyte conductivity employing stefan-maxwell
#

from .base_leading_surface_form_stefan_maxwell_conductivity import BaseLeadingOrderModel


class LeadingOrderModel(BaseLeadingOrderModel):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseLeadingSurfaceFormStefanMaxwellConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):

        if self._domain == "Separator":
            return {}

        j = variables[self._domain + " electrode interfacial current density"]
        j_av = variables[
            "Average " + self._domain.lower() + " electrode interfacial current density"
        ]
        delta_phi = variables[self._domain + " electrode surface potential difference"]

        self.algebraic[delta_phi] = j_av - j
