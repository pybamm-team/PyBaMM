#
# Base class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_surface_form_stefan_maxwell_conductivity import BaseModel


class BaseFullModel(BaseModel):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseModel`
    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_fundamental_variables(self):
        if self._domain == "Negative":
            delta_phi = pybamm.standard_variables.delta_phi_n
        elif self._domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p
        else:
            raise pybamm.DomainError

        delta_phi_av = pybamm.average(delta_phi)

        variables = pybamm.get_standard_surface_potential_difference_variables(
            delta_phi, delta_phi_av
        )

        return variables

    def set_initial_conditions(self, variables):
        delta_phi_e = variables[
            self._domain + " electrode surface potential difference"
        ]
        if self._domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init)
        elif self._domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init)

        else:
            raise pybamm.DomainError

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}

