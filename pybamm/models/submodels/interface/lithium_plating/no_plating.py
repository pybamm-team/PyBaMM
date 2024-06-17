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
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options=None, phase="primary"):
        super().__init__(param, domain, options=options, phase=phase)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), f"{self.domain} electrode", "current collector"
        )
        variables = self._get_standard_concentration_variables(zero, zero)
        variables.update(self._get_standard_overpotential_variables(zero))
        variables.update(self._get_standard_reaction_variables(zero))
        return variables

    def get_coupled_variables(self, variables):
        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables
