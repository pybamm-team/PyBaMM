#
# Class for composite surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from pybamm.models.submodels.electrolyte_conductivity.composite_conductivity import (
    Composite,
)


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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def get_fundamental_variables(self):
        if self.domain == "separator":
            return {}

        delta_phi_av = pybamm.Variable(
            f"X-averaged {self.domain} electrode surface potential difference [V]",
            domain="current collector",
            reference=self.domain_param.prim.U_init,
        )

        variables = self._get_standard_average_surface_potential_difference_variables(
            delta_phi_av
        )
        return variables

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        # Only update coupled variables once
        if self.domain == "negative":
            variables.update(super().get_coupled_variables(variables))

        phi_s = variables[f"{Domain} electrode potential [V]"]
        phi_e = variables[f"{Domain} electrolyte potential [V]"]
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        return variables

    def set_initial_conditions(self, variables):
        domain = self.domain

        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_boundary_conditions(self, variables):
        if self.domain == "negative":
            phi_e = variables["Electrolyte potential [V]"]
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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_rhs(self, variables):
        domain = self.domain

        a = variables[
            f"X-averaged {domain} electrode surface area to volume ratio [m-1]"
        ]

        sum_a_j = variables[
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities [A.m-3]"
        ]

        sum_a_j_av = variables[
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density [A.m-3]"
        ]
        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]

        T = variables[f"X-averaged {domain} electrode temperature [K]"]

        C_dl = self.domain_param.C_dl(T)

        self.rhs[delta_phi] = 1 / (a * C_dl) * (sum_a_j_av - sum_a_j)


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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_algebraic(self, variables):
        domain = self.domain

        sum_a_j = variables[
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities [A.m-3]"
        ]

        sum_a_j_av = variables[
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density [A.m-3]"
        ]
        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]

        self.algebraic[delta_phi] = (sum_a_j_av - sum_a_j) / self.param.a_j_scale
