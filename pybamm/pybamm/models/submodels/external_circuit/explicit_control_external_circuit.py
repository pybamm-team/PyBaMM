#
# External circuit with explicit equations for control
#
import pybamm
from .base_external_circuit import BaseModel


class ExplicitCurrentControl(BaseModel):
    """External circuit with current control."""

    def __init__(self, param, options):
        super().__init__(param, options)

    def get_fundamental_variables(self):
        # Current is given as a function of time
        i_cell = self.param.current_density_with_time
        I = self.param.current_with_time

        variables = {
            "Current variable [A]": I,
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / self.param.Q,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables


class ExplicitPowerControl(BaseModel):
    """External circuit with current set explicitly to hit target power."""

    def __init__(self, param, options):
        super().__init__(param, options)

    def get_coupled_variables(self, variables):
        param = self.param

        # Current is given as applied power divided by voltage
        V = variables["Voltage [V]"]
        P = pybamm.FunctionParameter("Power function [W]", {"Time [s]": pybamm.t})
        I = P / V

        # Update derived variables
        i_cell = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        return variables


class ExplicitResistanceControl(BaseModel):
    """External circuit with current set explicitly to hit target resistance."""

    def __init__(self, param, options):
        super().__init__(param, options)

    def get_coupled_variables(self, variables):
        param = self.param

        # Current is given as applied voltage divided by resistance
        V = variables["Voltage [V]"]
        R = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        I = V / R

        # Update derived variables
        i_cell = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        return variables
