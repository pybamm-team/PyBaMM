#
# Class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_leading_surface_form_stefan_maxwell_conductivity import BaseLeadingOrder


class LeadingOrderCapacitance(BaseLeadingOrder):
    """Leading-order model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation and where capacitance is present.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseLeadingOrder`

    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_rhs(self, variables):

        if self.domain == "Separator":
            return None

        param = self.param

        j = variables[self.domain + " electrode interfacial current density"]
        j_av = variables[
            "Average " + self.domain.lower() + " electrode interfacial current density"
        ]
        delta_phi = variables[self.domain + " electrode surface potential difference"]

        if self.domain == "Negative":
            C_dl = param.C_dl_n
        elif self.domain == "Positive":
            C_dl = param.C_dl_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self.domain))

        self.rhs[delta_phi] = 1 / C_dl * (j_av - j)
