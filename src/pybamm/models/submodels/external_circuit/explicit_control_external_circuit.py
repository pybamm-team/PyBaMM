#
# External circuit with explicit equations for control
#
import pybamm
from .base_external_circuit import BaseModel


class ExplicitCurrentControl(BaseModel):
    """External circuit with current control."""

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
        if self.options.get("voltage as a state") == "true":
            V = pybamm.Variable("Voltage [V]")
            variables.update({"Voltage [V]": V})

        return variables

    def set_initial_conditions(self, variables):
        if self.options.get("voltage as a state") == "true":
            V = variables["Voltage [V]"]
            self.initial_conditions[V] = self.param.ocv_init

    def set_algebraic(self, variables):
        if self.options.get("voltage as a state") == "true":
            V = variables["Voltage [V]"]
            V_expression = variables["Voltage expression [V]"]
            self.algebraic[V] = V - V_expression


class ExplicitPowerControl(BaseModel):
    """External circuit with current set explicitly to hit target power."""

    def get_coupled_variables(self, variables):
        # Current is given as applied power divided by voltage
        V = variables["Voltage [V]"]
        P = pybamm.FunctionParameter("Power function [W]", {"Time [s]": pybamm.t})
        I = P / V

        # Update derived variables
        i_cell = I / (self.param.n_electrodes_parallel * self.param.A_cc)

        variables = {
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / self.param.Q,
        }

        return variables


class ExplicitResistanceControl(BaseModel):
    """External circuit with current set explicitly to hit target resistance."""

    def get_coupled_variables(self, variables):
        # Current is given as applied voltage divided by resistance
        V = variables["Voltage [V]"]
        R = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        I = V / R

        # Update derived variables
        i_cell = I / (self.param.n_electrodes_parallel * self.param.A_cc)

        variables = {
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / self.param.Q,
        }

        return variables
