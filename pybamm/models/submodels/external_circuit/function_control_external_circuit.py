#
# External circuit with an arbitrary function
#
import pybamm
from .base_external_circuit import BaseModel


class FunctionControl(BaseModel):
    """
    External circuit with an arbitrary function, implemented as a control on the current
    either via an algebraic equation, or a differential equation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    external_circuit_function : callable
        The function that controls the current
    options : dict
        Dictionary of options to use for the submodel
    control : str, optional
        The type of control to use. Must be one of 'algebraic' (default)
        or 'differential'.
    """

    def __init__(self, param, external_circuit_function, options, control="algebraic"):
        super().__init__(param, options)
        self.external_circuit_function = external_circuit_function
        self.control = control

    def get_fundamental_variables(self):
        param = self.param
        # Current is a variable
        i_var = pybamm.Variable("Current variable [A]", scale=param.Q)
        if self.control in ["algebraic", "differential without max"]:
            I = i_var
        elif self.control == "differential with max":
            i_input = pybamm.FunctionParameter(
                "CCCV current function [A]", {"Time [s]": pybamm.t}
            )
            I = pybamm.maximum(i_var, i_input)

        # Update derived variables
        i_cell = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Current variable [A]": i_var,
            "Total current density [A.m-2]": i_cell,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        # Initial condition as a guess for consistent initial conditions
        i_cell = variables["Current variable [A]"]
        self.initial_conditions[i_cell] = self.param.Q

    def set_rhs(self, variables):
        super().set_rhs(variables)
        # External circuit submodels are always equations on the current
        # The external circuit function should provide an update law for the current
        # based on current/voltage/power/etc.
        if "differential" in self.control:
            i_cell = variables["Current variable [A]"]
            self.rhs[i_cell] = self.external_circuit_function(variables)

    def set_algebraic(self, variables):
        # External circuit submodels are always equations on the current
        # The external circuit function should fix either the current, or the voltage,
        # or a combination (e.g. I*V for power control)
        if self.control == "algebraic":
            i_cell = variables["Current variable [A]"]
            self.algebraic[i_cell] = self.external_circuit_function(variables)


class VoltageFunctionControl(FunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation.
    """

    def __init__(self, param, options):
        super().__init__(param, self.constant_voltage, options, control="algebraic")

    def constant_voltage(self, variables):
        V = variables["Voltage [V]"]
        return V - pybamm.FunctionParameter(
            "Voltage function [V]", {"Time [s]": pybamm.t}
        )


class PowerFunctionControl(FunctionControl):
    """External circuit with power control."""

    def __init__(self, param, options, control="algebraic"):
        super().__init__(param, self.constant_power, options, control=control)

    def constant_power(self, variables):
        I = variables["Current [A]"]
        V = variables["Voltage [V]"]
        P = V * I
        P_applied = pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t}
        )
        if self.control == "algebraic":
            return P - P_applied
        else:
            # Multiply by the time scale so that the overshoot only lasts a few seconds
            K_P = 0.01
            return -K_P * (P - P_applied)


class ResistanceFunctionControl(FunctionControl):
    """External circuit with resistance control."""

    def __init__(self, param, options, control):
        super().__init__(param, self.constant_resistance, options, control=control)

    def constant_resistance(self, variables):
        I = variables["Current [A]"]
        V = variables["Voltage [V]"]
        R = V / I
        R_applied = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        if self.control == "algebraic":
            return R - R_applied
        else:
            # Multiply by the time scale so that the overshoot only lasts a few seconds
            K_R = 0.01
            return -K_R * (R - R_applied)


class CCCVFunctionControl(FunctionControl):
    """
    External circuit with constant-current constant-voltage control, as implemented in
    :footcite:t:`Mohtat2021`.

    .. footbibliography::

    """

    def __init__(self, param, options):
        super().__init__(param, self.cccv, options, control="differential with max")
        pybamm.citations.register("Mohtat2021")

    def cccv(self, variables):
        # Multiply by the time scale so that the votage overshoot only lasts a few
        # seconds
        K_aw = 1  # anti-windup
        Q = self.param.Q
        I_var = variables["Current variable [A]"]
        I = variables["Current [A]"]

        K_V = 1
        V = variables["Voltage [V]"]
        V_CCCV = pybamm.Parameter("Voltage function [V]")

        return -K_aw / Q * (I_var - I) + K_V * (V - V_CCCV)
