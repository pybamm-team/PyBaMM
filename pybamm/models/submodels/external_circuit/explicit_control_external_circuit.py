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
        i_cell = self.param.current_with_time
        i_cell_dim = self.param.dimensional_current_density_with_time
        I = self.param.dimensional_current_with_time

        variables = {
            "Current density variable": pybamm.Scalar(1, name="i_cell"),
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
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
        V = variables["Terminal voltage [V]"]
        P = pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t * self.param.timescale}
        )
        I = P / V

        # Update derived variables
        i_cell = I / abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
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
        V = variables["Terminal voltage [V]"]
        R = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t * self.param.timescale}
        )
        I = V / R

        # Update derived variables
        i_cell = I / abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        return variables
