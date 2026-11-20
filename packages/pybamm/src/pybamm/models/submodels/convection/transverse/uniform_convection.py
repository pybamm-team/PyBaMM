#
# Submodel for uniform convection in transverse directions
#
import pybamm

from .base_transverse_convection import BaseTransverseModel


class Uniform(BaseTransverseModel):
    """
    Submodel for uniform convection in transverse directions

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    """

    def __init__(self, param):
        super().__init__(param)

    def build(self):
        p_s = pybamm.PrimaryBroadcast(0, "current collector")
        self.variables.update(self._get_standard_separator_pressure_variables(p_s))

        div_Vbox_s = self._get_separator_acceleration()
        self.variables.update(
            self._get_standard_transverse_velocity_variables(div_Vbox_s, "acceleration")
        )

        z = pybamm.standard_spatial_vars.z
        Vbox_s = pybamm.IndefiniteIntegral(div_Vbox_s, z)
        self.variables.update(
            self._get_standard_transverse_velocity_variables(Vbox_s, "velocity")
        )
