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

    def build(self):
        domain = self.domain
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
        self.variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi_av)
        )
        # The negative electrode submodel also sets the electrolyte potential
        if domain == "negative":
            self._build_leading_order(delta_phi_av)
        self.initial_conditions[delta_phi_av] = self.domain_param.prim.U_init


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

    def build(self):
        super().build()
        delta_phi_av, current_imbalance = self._get_x_averaged_current_imbalance()
        self.rhs[delta_phi_av] = self._get_x_averaged_double_layer_rhs(
            current_imbalance
        )


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

    def build(self):
        super().build()
        delta_phi_av, current_imbalance = self._get_x_averaged_current_imbalance()
        # multiply by Lx**2 to improve conditioning
        self.algebraic[delta_phi_av] = current_imbalance * self.param.L_x**2
