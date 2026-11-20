#
# External circuit with an arbitrary function
#
import pybamm

from .base_external_circuit import BaseModel


class _ControlVariables:
    """
    The variables available to the function of an external circuit submodel.

    Variables the submodel defines (e.g. "Current [A]") are returned as they are.
    Any other name returns a :class:`pybamm.CoupledVariable` with no domain, which
    is replaced by the model variable of that name when the model is built.

    Parameters
    ----------
    submodel : :class:`pybamm.external_circuit.BaseModel`
        The external circuit submodel calling the function.
    """

    def __init__(self, submodel):
        self._submodel = submodel

    def __getitem__(self, name):
        if name in self._submodel.variables:
            return self._submodel.variables[name]
        return self._submodel.coupled_variable(name)


class FunctionControl(BaseModel):
    """
    External circuit with an arbitrary function, implemented as a control on the current
    either via an algebraic equation, or a differential equation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    external_circuit_function : callable
        The function that controls the current. It is called with a mapping from
        variable names to symbols; variables from other submodels are taken to have
        no domain.
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

    def build(self):
        i_var = pybamm.Variable("Current variable [A]", scale=self.param.Q)
        if self.control in ["algebraic", "differential"]:
            I = i_var
        elif self.control == "differential with max":
            i_input = pybamm.FunctionParameter(
                "CCCV current function [A]", {"Time [s]": pybamm.t}
            )
            I = pybamm.maximum(i_var, i_input)
        self.variables.update(
            {"Current variable [A]": i_var, **self._get_current_variables(I)}
        )
        # Initial condition as a guess for consistent initial conditions
        self.initial_conditions[i_var] = self.param.Q

        # The function gives either an update law for the current (differential) or
        # a residual fixing the current, voltage, power, etc. (algebraic)
        residual = self.external_circuit_function(_ControlVariables(self))
        if self.control == "algebraic":
            self.algebraic[i_var] = residual
        else:
            self.rhs[i_var] = residual


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

    def __init__(self, param, options, control="algebraic"):
        super().__init__(param, self.constant_resistance, options, control=control)

    def constant_resistance(self, variables):
        I = variables["Current [A]"]
        V = variables["Voltage [V]"]
        R_applied = pybamm.FunctionParameter(
            "Resistance function [Ohm]", {"Time [s]": pybamm.t}
        )
        if self.control == "algebraic":
            return V - R_applied * I
        else:
            # Multiply by the time scale so that the overshoot only lasts a few seconds
            K_R = 0.01
            return -K_R * (V - R_applied * I)


class ExperimentFunctionControl(FunctionControl):
    """External circuit with one conditional control residual shared across steps."""

    def __init__(self, param, options, step_index_input_name, step_control_builders):
        self.step_index_input_name = step_index_input_name
        self.step_control_builders = step_control_builders
        super().__init__(
            param,
            self.conditional_control_residual,
            options,
            control="algebraic",
        )

    def conditional_control_residual(self, variables):
        selector = pybamm.InputParameter(self.step_index_input_name)
        return pybamm.Conditional(
            selector,
            *[builder(variables) for builder in self.step_control_builders],
        )


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
