#
# External circuit with explicit equations for control
#
import pybamm

from .base_external_circuit import BaseModel


class ExplicitCurrentControl(BaseModel):
    """External circuit with current control."""

    def build(self):
        # Current is given as a function of time
        I = self.param.current_with_time
        self.variables.update(
            {
                "Current variable [A]": I,
                "Total current density [A.m-2]": self.param.current_density_with_time,
                "Current [A]": I,
                "C-rate": I / self.param.Q,
            }
        )


class ExplicitPowerControl(BaseModel):
    """External circuit with current set explicitly to hit target power."""

    def build(self):
        # Current is given as applied power divided by voltage
        V = self.coupled_variable("Voltage [V]")
        P = pybamm.FunctionParameter("Power function [W]", {"Time [s]": pybamm.t})
        I = P / V
        self.variables.update(self._get_current_variables(I))


class ExplicitResistanceControl(BaseModel):
    """External circuit with current set explicitly to hit target resistance."""

    def build(self):
        # Current is given as applied voltage divided by resistance
        V = self.coupled_variable("Voltage [V]")
        R = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        I = V / R
        self.variables.update(self._get_current_variables(I))
