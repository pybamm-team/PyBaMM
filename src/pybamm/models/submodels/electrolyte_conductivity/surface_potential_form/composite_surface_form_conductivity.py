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

    def _get_fundamental_variables(self):
        if self.domain == "separator":
            return

        delta_phi_av = pybamm.Variable(
            f"X-averaged {self.domain} electrode surface potential difference [V]",
            domain="current collector",
            reference=self.domain_param.prim.U_init,
        )

        variables = self._get_standard_average_surface_potential_difference_variables(
            delta_phi_av
        )
        return variables

    def _set_initial_conditions(self, variables):
        domain = self.domain

        delta_phi = variables[
            f"X-averaged {domain} electrode surface potential difference [V]"
        ]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def _get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        # Only update coupled variables once
        if self.domain == "negative":
            variables.update(super()._get_coupled_variables(variables))

        phi_s = pybamm.CoupledVariable(
            f"{Domain} electrode potential [V]",
            domain=f"{self.domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({phi_s.name: phi_s})
        phi_e = pybamm.CoupledVariable(
            f"{Domain} electrolyte potential [V]",
            domain=f"{self.domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({phi_e.name: phi_e})
        delta_phi = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}
        if self.domain == "negative":
            phi_e = variables["Electrolyte potential [V]"]
            self.boundary_conditions = {
                phi_e: {
                    "left": (pybamm.Scalar(0), "Neumann"),
                    "right": (pybamm.Scalar(0), "Neumann"),
                }
            }
        return variables


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

    def _set_rhs(self, variables):
        domain = self.domain

        a = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode surface area to volume ratio [m-1]",
            domain="current collector",
        )
        self.coupled_variables.update({a.name: a})

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
        T = pybamm.CoupledVariable(
            f"X-averaged {domain} electrode temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T.name: T})

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

    def build(self, submodels):
        domain = self.domain

        variables = self._get_fundamental_variables()
        variables.update(self._get_coupled_variables(variables))

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

        delta_phi = pybamm.Variable(
            f"X-averaged {domain} electrode surface potential difference [V]",
            domain="current collector",
        )
        self.coupled_variables.update({delta_phi.name: delta_phi})

        self.algebraic[delta_phi] = (sum_a_j_av - sum_a_j) / self.param.a_j_scale
        self.variables.update(variables)
