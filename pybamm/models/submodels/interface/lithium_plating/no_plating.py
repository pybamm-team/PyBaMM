#
# Class for no plating
#
import pybamm
from .base_plating import BasePlating


class NoPlating(BasePlating):
    """Base class for no lithium plating/stripping.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.lithium_plating.BasePlating`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower() + " electrode", "current collector"
        )
        variables = self._get_standard_concentration_variables(zero)
        variables.update(self._get_standard_reaction_variables(zero))
        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode lithium plating interfacial current density"
            in variables
            and "Positive electrode lithium plating interfacial current density"
            in variables
            and "Lithium plating interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables
