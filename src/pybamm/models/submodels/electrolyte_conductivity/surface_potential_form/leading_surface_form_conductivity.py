#
# Class for leading-order surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from pybamm.models.submodels.electrolyte_conductivity.leading_order_conductivity import (
    LeadingOrder,
)


class BaseLeadingOrderSurfaceForm(LeadingOrder):
    """Base class for leading-order conservation of charge in the electrolyte employing
    the Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Leading refers to leading order in asymptotics)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def build(self, submodels):
        delta_phi_av = pybamm.Variable(
            f"X-averaged {self.domain} electrode surface potential difference [V]",
            domain="current collector",
            reference=self.domain_param.prim.U_init,
        )

        variables = self._get_standard_average_surface_potential_difference_variables(
            delta_phi_av
        )
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi_av)
        )
        # Only update coupled variables once
        if self.domain == "negative":
            super().build(submodels)
        delta_phi = delta_phi_av
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}
        if self.domain == "negative":
            phi_e = self.variables["Electrolyte potential [V]"]
            self.boundary_conditions = {
                phi_e: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),
                }
            }
        self._set_eqn(variables)
        self.variables.update(variables)


class LeadingOrderDifferential(BaseLeadingOrderSurfaceForm):
    """Leading-order model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation and where capacitance is present.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def _set_eqn(self, variables):
        domain = self.domain
        T = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T.name: T})
        C_dl = self.domain_param.C_dl(T)

        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]

        sum_a_j = pybamm.CoupledVariable(
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_a_j.name: sum_a_j})
        sum_a_j_av = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_a_j_av.name: sum_a_j_av})
        a = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode surface area to volume ratio [m-1]",
            domain="current collector",
        )
        self.coupled_variables.update({a.name: a})

        self.rhs[delta_phi] = 1 / (a * C_dl) * (sum_a_j_av - sum_a_j)


class LeadingOrderAlgebraic(BaseLeadingOrderSurfaceForm):
    """Leading-order model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def _set_eqn(self, variables):
        domain = self.domain
        sum_a_j = pybamm.CoupledVariable(
            f"Sum of x-averaged {domain} electrode volumetric "
            "interfacial current densities [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_a_j.name: sum_a_j})

        sum_a_j_av = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode total volumetric "
            "interfacial current density [A.m-3]",
            domain="current collector",
        )
        self.coupled_variables.update({sum_a_j_av.name: sum_a_j_av})
        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]

        # multiply by Lx**2 to improve conditioning
        self.algebraic[delta_phi] = (sum_a_j_av - sum_a_j) * self.param.L_x**2
