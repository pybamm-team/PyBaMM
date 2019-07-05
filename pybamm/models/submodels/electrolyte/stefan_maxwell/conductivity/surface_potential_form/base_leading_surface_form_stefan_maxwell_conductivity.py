#
# Base class for leading surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm
from .base_surface_form_stefan_maxwell_conductivity import BaseModel


class BaseLeadingOrder(BaseModel):
    """Base class for leading-order conservation of charge in the electrolyte employing the Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Leading refers to leading order in asymptotics)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseModel`
    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):

        if self.domain == "Negative":
            delta_phi = pybamm.standard_variables.delta_phi_n_av
        elif self.domain == "Separator":
            return {}
        elif self.domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p_av
        else:
            raise pybamm.DomainError

        variables = self._get_standard_surface_potential_difference_variables(delta_phi)
        return variables

    def set_initial_conditions(self, variables):

        if self.domain == "Separator":
            return {}

        delta_phi_e = variables[self.domain + " electrode surface potential difference"]
        if self.domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init)
        elif self.domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init)

        else:
            raise pybamm.DomainError

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}
