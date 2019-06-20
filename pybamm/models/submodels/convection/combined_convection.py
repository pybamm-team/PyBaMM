#
# Class for combined leading and first order pressure driven convection
#
import pybamm
from .base_convection import BaseModel


class CombinedOrder(BaseModel):
    """Class for combined leading and first-order pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):
        param = self.param
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.IndefiniteIntegral(j_n, x_n)
        # Shift v_box_p to be equal to 0 at x_p = 1
        v_box_p = param.beta_p * (
            pybamm.IndefiniteIntegral(j_p, x_p) - pybamm.Integral(j_p, x_p)
        )

        v_box_s, dVbox_dz = self.get_separator_velocities(variables)
        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)

        variables.update(self._get_standard_velocity_variables(v_box))
        variables.update(self._get_standard_vertical_velocity_variables(dVbox_dz))

        return variables

