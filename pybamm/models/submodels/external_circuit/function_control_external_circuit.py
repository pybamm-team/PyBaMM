#
# External circuit with an arbitrary function
#
import pybamm
from .base_external_circuit import BaseModel, LeadingOrderBaseModel


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
    control : str, optional
        The type of control to use. Must be one of 'algebraic' (default)
        or 'differential'.
    """

    def __init__(self, param, external_circuit_function, control="algebraic"):
        super().__init__(param)
        self.external_circuit_function = external_circuit_function
        self.control = control

    def get_fundamental_variables(self):
        param = self.param
        # Current is a variable
        i_var = pybamm.Variable("Current density variable")
        if self.control == "algebraic":
            i_cell = i_var
        elif self.control == "differential":
            i_cell = pybamm.maximum(i_var, param.current_with_time)

        # Update derived variables
        I = i_cell * abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Current density variable": i_var,
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        # Add discharge capacity variable
        variables.update(super().get_fundamental_variables())

        return variables

    def set_initial_conditions(self, variables):
        super().set_initial_conditions(variables)
        # Initial condition as a guess for consistent initial conditions
        i_cell = variables["Current density variable"]
        self.initial_conditions[i_cell] = self.param.current_with_time

    def set_rhs(self, variables):
        super().set_rhs(variables)
        # External circuit submodels are always equations on the current
        # The external circuit function should provide an update law for the current
        # based on current/voltage/power/etc.
        if self.control == "differential":
            i_cell = variables["Current density variable"]
            self.rhs[i_cell] = self.external_circuit_function(variables)

    def set_algebraic(self, variables):
        # External circuit submodels are always equations on the current
        # The external circuit function should fix either the current, or the voltage,
        # or a combination (e.g. I*V for power control)
        if self.control == "algebraic":
            i_cell = variables["Current density variable"]
            self.algebraic[i_cell] = self.external_circuit_function(variables)


class VoltageFunctionControl(FunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation.
    """

    def __init__(self, param):
        super().__init__(param, self.constant_voltage, control="algebraic")

    def constant_voltage(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter(
            "Voltage function [V]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class PowerFunctionControl(FunctionControl):
    """External circuit with power control."""

    def __init__(self, param):
        super().__init__(param, self.constant_power, control="algebraic")

    def constant_power(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class CCCVFunctionControl(FunctionControl):
    """
    External circuit with constant-current constant-voltage control, as implemented in
    [1]_

    References
    ----------
    .. [1] Mohtat, P., Pannala, S., Sulzer, V., Siegel, J. B., & Stefanopoulou, A. G.
           (2021). An Algorithmic Safety VEST For Li-ion Batteries During Fast Charging.
           arXiv preprint arXiv:2108.07833.

    """

    def __init__(self, param):
        super().__init__(param, self.cccv, control="differential")
        pybamm.citations.register("Mohtat2021")

    def cccv(self, variables):
        # Multiply by the time scale so that the votage overshoot only lasts a few
        # seconds
        K_aw = 1 * self.param.timescale  # anti-windup
        K_V = 1 * self.param.timescale
        i_var = variables["Current density variable"]
        i_cell = variables["Total current density"]
        V = variables["Terminal voltage [V]"]
        V_CCCV = pybamm.Parameter("Voltage function [V]")
        return -K_aw * (i_var - i_cell) + K_V * (V - V_CCCV)


class LeadingOrderFunctionControl(FunctionControl, LeadingOrderBaseModel):
    """External circuit with an arbitrary function, at leading order."""

    def __init__(self, param, external_circuit_class, control="algebraic"):
        super().__init__(param, external_circuit_class, control=control)

    def _get_current_variable(self):
        return pybamm.Variable("Leading-order total current density")


class LeadingOrderVoltageFunctionControl(LeadingOrderFunctionControl):
    """
    External circuit with voltage control, implemented as an extra algebraic equation,
    at leading order.
    """

    def __init__(self, param):
        super().__init__(param, self.constant_voltage, control="algebraic")

    def constant_voltage(self, variables):
        V = variables["Terminal voltage [V]"]
        return V - pybamm.FunctionParameter(
            "Voltage function [V]", {"Time [s]": pybamm.t * self.param.timescale}
        )


class LeadingOrderPowerFunctionControl(LeadingOrderFunctionControl):
    """External circuit with power control, at leading order."""

    def __init__(self, param):
        super().__init__(param, self.constant_power, control="algebraic")

    def constant_power(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I * V - pybamm.FunctionParameter(
            "Power function [W]", {"Time [s]": pybamm.t * self.param.timescale}
        )
