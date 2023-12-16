#
# Submodel for no convection in transverse directions
#
import pybamm
from .base_transverse_convection import BaseTransverseModel


class NoConvection(BaseTransverseModel):
    """
    Submodel for no convection in transverse directions

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        p_s = pybamm.PrimaryBroadcast(0, "current collector")
        variables = self._get_standard_separator_pressure_variables(p_s)

        Vbox_s = pybamm.PrimaryBroadcast(0, "current collector")
        variables.update(
            self._get_standard_transverse_velocity_variables(Vbox_s, "velocity")
        )

        div_Vbox_s = pybamm.PrimaryBroadcast(0, "current collector")

        variables.update(
            self._get_standard_transverse_velocity_variables(div_Vbox_s, "acceleration")
        )

        return variables
