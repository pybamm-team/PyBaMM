#
# External circuit with voltage control
#
import pybamm
from .base_external_circuit import BaseModel


class VoltageControl(BaseModel):
    """External circuit with voltage control. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Current is given as a function of time
        i_cell = pybamm.electrical_parameters.current_with_time
        i_cell_dim = pybamm.electrical_parameters.dimensional_current_density_with_time
        I = pybamm.electrical_parameters.dimensional_current_with_time

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "Discharge capacity [A.h]": I * pybamm.t * self.param.time_scale / 3600,
        }

        return variables
