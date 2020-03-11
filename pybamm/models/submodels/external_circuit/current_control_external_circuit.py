#
# External circuit with current control
#
from .base_external_circuit import BaseModel, LeadingOrderBaseModel


class CurrentControl(BaseModel):
    """External circuit with current control. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Current is given as a function of time
        i_cell = self.param.current_with_time
        i_cell_dim = self.param.dimensional_current_density_with_time
        I = self.param.dimensional_current_with_time

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "C-rate": I / self.param.Q,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables


class LeadingOrderCurrentControl(CurrentControl, LeadingOrderBaseModel):
    """External circuit with current control, for leading order models. """

    def __init__(self, param):
        super().__init__(param)
