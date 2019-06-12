#
# Class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm


class LeadingCapacitanceStefanMaxwellConductivity(
    pybamm.BaseLeadingSurfaceFormStefanMaxwellConductivity
):
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

    def set_differential(self, variables):

        param = self.param

        j = variables[self._domain + " electrode interfacial current density"]
        j_av = variables[
            "Average " + self._domain + " electrode interfacial current density"
        ]
        delta_phi = variables[self._domain + " electrode surface potential difference"]

        if self._domain == "Negative":
            C_dl = param.C_dl_n
        elif self._domain == "Positive":
            C_dl = param.C_dl_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self._domain))

        self.rhs[delta_phi] = 1 / C_dl * (j_av - j)
