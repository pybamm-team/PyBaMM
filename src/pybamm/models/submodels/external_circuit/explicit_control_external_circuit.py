#
# External circuit with explicit equations for control
#
import pybamm
from .base_external_circuit import BaseModel


class ExplicitCurrentControl(BaseModel):
    """External circuit with current control."""

    def build(self):
        i_cell = self.param.current_density_with_time
        I = self.param.current_with_time
        self.variables.update(
            {
                "Current variable [A]": I,
                "Total current density [A.m-2]": i_cell,
                "Current [A]": I,
                "C-rate": I / self.param.Q,
            }
        )
        if self.options.get("voltage as a state") == "true":
            V = pybamm.Variable("Voltage [V]")
            self.variables.update({"Voltage [V]": V})
            self.initial_conditions[V] = self.param.ocv_init
            V_expression = pybamm.CoupledVariable("Voltage expression [V]")
            self.coupled_variables.update({"Voltage expression [V]": V_expression})
            self.algebraic[V] = V - V_expression


class ExplicitPowerControl(BaseModel):
    """External circuit with current set explicitly to hit target power."""

    def build(self):
        V = pybamm.CoupledVariable("Voltage [V]")
        self.coupled_variables.update({"Voltage [V]": V})
        P = pybamm.FunctionParameter("Power function [W]", {"Time [s]": pybamm.t})
        I = P / V
        i_cell = I / (self.param.n_electrodes_parallel * self.param.A_cc)
        self.variables.update(
            {
                "Total current density [A.m-2]": i_cell,
                "Current [A]": I,
                "C-rate": I / self.param.Q,
            }
        )

    def get_coupled_variables_LEGACY(self, variables):
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

    def build(self):
        V = pybamm.CoupledVariable("Voltage [V]")
        self.coupled_variables.update({"Voltage [V]": V})

        R = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        I = V / R
        i_cell = I / (self.param.n_electrodes_parallel * self.param.A_cc)
        self.variables.update(
            {
                "Total current density [A.m-2]": i_cell,
                "Current [A]": I,
                "C-rate": I / self.param.Q,
            }
        )

    def get_coupled_variables_LEGACY(self, variables):
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
