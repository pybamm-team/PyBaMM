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

    **Extends:** :class:`pybamm.lithium_plating.BasePlating`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), "negative electrode", "current collector"
        )
        variables = self._get_standard_concentration_variables(zero)
        variables.update(self._get_standard_reaction_variables(zero))
        return variables

    def get_coupled_variables(self, variables):
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), "negative electrode", "current collector"
        )
        variables.update(self._get_standard_overpotential_variables(zero))
        variables.update(self._get_standard_reaction_variables(zero))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables
