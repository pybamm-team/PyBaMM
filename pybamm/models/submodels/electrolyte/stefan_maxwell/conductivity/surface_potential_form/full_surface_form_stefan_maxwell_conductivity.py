#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm
from .base_full_surface_form_stefan_maxwell_conductivity import BaseFullModel


class FullModel(BaseFullModel):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


     **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseFullModel`
    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):
        delta_phi = variables[self._domain + " electrode surface potential difference"]
        i_e = variables[self._domain + " electrolyte current density"]
        j = variables[self._domain + " electrode interfacial current density"]

        self.rhs[delta_phi] = pybamm.div(i_e) - j

