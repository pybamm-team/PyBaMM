#
# Class for composite surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from ..composite_conductivity import Composite


class BaseModel(Composite):
    """
    Base class for composite conservation of charge in the electrolyte employing
    the Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`pybamm.electrolyte_conductivity.Composite`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def get_fundamental_variables(self):
        if self.domain == "negative":
            delta_phi_av = pybamm.standard_variables.delta_phi_n_av
        elif self.domain == "separator":
            return {}
        elif self.domain == "positive":
            delta_phi_av = pybamm.standard_variables.delta_phi_p_av

        variables = self._get_standard_average_surface_potential_difference_variables(
            delta_phi_av
        )
        return variables

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        # Only update coupled variables once
        if self.domain == "negative":
            variables.update(super().get_coupled_variables(variables))

        phi_s = variables[f"{Domain} electrode potential"]
        phi_e = variables[f"{Domain} electrolyte potential"]
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        return variables

    def set_initial_conditions(self, variables):
        domain = self.domain

        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference"
        ]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_boundary_conditions(self, variables):
        if self.domain == "negative":
            phi_e = variables["Electrolyte potential"]
            self.boundary_conditions = {
                phi_e: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),
                }
            }


class CompositeDifferential(BaseModel):
    """
    Composite model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation and where capacitance is present.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`BaseModel`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_rhs(self, variables):
        domain = self.domain

        sum_a_j = variables[
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities"
        ]

        sum_a_j_av = variables[
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density"
        ]
        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference"
        ]

        C_dl = self.domain_param.C_dl

        self.rhs[delta_phi] = 1 / C_dl * (sum_a_j_av - sum_a_j)


class CompositeAlgebraic(BaseModel):
    """
    Composite model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict
        Additional options to pass to the model

    **Extends:** :class:`BaseModel`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_algebraic(self, variables):
        domain = self.domain

        sum_a_j = variables[
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities"
        ]

        sum_a_j_av = variables[
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density"
        ]
        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference"
        ]

        self.algebraic[delta_phi] = sum_a_j_av - sum_a_j
