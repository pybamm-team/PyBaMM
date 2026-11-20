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

    def build(self):
        domain, Domain = self.domain_Domain
        delta_phi_av = pybamm.Variable(
            f"X-averaged {domain} electrode surface potential difference [V]",
            domain="current collector",
            reference=self.domain_param.prim.U_init,
        )
        self.variables.update(
            self._get_standard_average_surface_potential_difference_variables(
                delta_phi_av
            )
        )
        self.initial_conditions[delta_phi_av] = self.domain_param.prim.U_init

        # The negative electrode submodel also sets the electrolyte potential
        if domain == "negative":
            self._build_composite(delta_phi_av)
            phi_e = self.variables["Negative electrolyte potential [V]"]
        else:
            phi_e = self._coupled_domain_variable(
                f"{Domain} electrolyte potential [V]", f"{domain} electrode"
            )
        phi_s = self._coupled_domain_variable(
            f"{Domain} electrode potential [V]", f"{domain} electrode"
        )
        self.variables.update(
            self._get_standard_surface_potential_difference_variables(phi_s - phi_e)
        )


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

    def build(self):
        super().build()
        delta_phi_av, current_imbalance = self._get_x_averaged_current_imbalance()
        self.rhs[delta_phi_av] = self._get_x_averaged_double_layer_rhs(
            current_imbalance
        )


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

    def build(self):
        super().build()
        delta_phi_av, current_imbalance = self._get_x_averaged_current_imbalance()
        self.algebraic[delta_phi_av] = current_imbalance / self.param.a_j_scale
