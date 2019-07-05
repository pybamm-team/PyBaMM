#
# Class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm
from .base_surface_form_stefan_maxwell_conductivity import BaseModel


class BaseLeadingOrder(BaseModel):
    """Base class for leading-order conservation of charge in the electrolyte employing
    the Stefan-Maxwell constitutive equations employing the surface potential difference
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
            delta_phi = pybamm.standard_variables.delta_phi_n
        elif self.domain == "Separator":
            return {}
        elif self.domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p
        else:
            raise pybamm.DomainError

        delta_phi_av = pybamm.average(delta_phi)

        variables = self._get_standard_surface_potential_difference_variables(
            delta_phi, delta_phi_av
        )
        return variables

    def set_initial_conditions(self, variables):

        if self.domain == "Separator":
            return

        delta_phi_e = variables[self.domain + " electrode surface potential difference"]
        if self.domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init)
        elif self.domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init)

        else:
            raise pybamm.DomainError

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}


class LeadingOrderDifferential(BaseLeadingOrder):
    """Leading-order model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation and where capacitance is present.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`BaseLeadingOrder`

    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_rhs(self, variables):
        if self.domain == "Separator":
            return

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

        self.rhs[delta_phi] = 1 / C_dl * (j_av - j)


class LeadingOrderAlgebraic(BaseLeadingOrder):
    """Leading-order model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`BaseLeadingOrder`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):
        if self.domain == "Separator":
            return

        j = variables[self.domain + " electrode interfacial current density"]
        j_av = variables[
            "Average " + self.domain.lower() + " electrode interfacial current density"
        ]
        delta_phi = variables[self.domain + " electrode surface potential difference"]

        self.algebraic[delta_phi] = j_av - j
