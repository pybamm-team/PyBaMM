#
# Class for leading-order pressure driven convection
#
import pybamm
from .base_convection import BaseModel


class LeadingOrder(BaseModel):
    """A submodel for the leading-order approximation of pressure-driven convection

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.convection.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):

        param = self.param
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        j_n_av = variables["X-averaged negative electrode interfacial current density"]
        j_p_av = variables["X-averaged positive electrode interfacial current density"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * j_n_av * x_n
        v_box_p = param.beta_p * j_p_av * (x_p - 1)

        v_box_s, dVbox_dz = self._separator_velocity(variables)
        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)

        variables.update(self._get_standard_velocity_variables(v_box))
        variables.update(self._get_standard_vertical_velocity_variables(dVbox_dz))

        return variables
