import copy

import numpy as np

import pybamm

from .base_step import (
    BaseStep,
    BaseStepExplicit,
    BaseStepImplicit,
    ControlKind,
    _adjusted_control_variable,
    _check_drive_cycle,
    _check_function_of_time,
    _constant_value,
    _control_variable,
    _convert_electric,
    _default_control_value,
    _drive_cycle_interpolant,
    _examples,
    _initial_control_guess,
    _step_time,
    _value_identity,
)


def string(text, **kwargs):
    """
    Create a step from a string.

    Parameters
    ----------
    text : str
        The string to parse. Each operating condition should
        be of the form "Do this for this long" or "Do this until this happens". For
        example, "Charge at 1 C for 1 hour", or "Charge at 1 C until 4.2 V", or "Charge
        at 1 C for 1 hour or until 4.2 V". The instructions can be of the form
        "(Dis)charge at x A/C/W", "Rest", or "Hold at x V until y A". The running
        time should be a time in seconds, minutes or hours, e.g. "10 seconds",
        "3 minutes" or "1 hour". The stopping conditions should be
        a circuit state, e.g. "1 A", "C/50" or "3 V".
    **kwargs
        Any other keyword arguments are passed to the step class

    Returns
    -------
    :class:`pybamm.step.BaseStep`
        A step parsed from the string.
    """
    if not isinstance(text, str):
        raise TypeError("Input to step.string() must be a string")

    if "oC" in text:
        raise ValueError(
            "Temperature must be specified as a keyword argument "
            "instead of in the string"
        )

    # Save the original string
    description = text

    # extract period
    if "period)" in text:
        if "period" in kwargs:
            raise ValueError(
                "Period cannot be specified both as a keyword argument "
                "and in the string"
            )
        text, period_full = text.split(" (")
        period, _ = period_full.split(" period)")
        kwargs["period"] = period
    # extract termination condition based on "until" keyword
    if "until" in text:
        # e.g. "Charge at 4 A until 3.8 V"
        text, termination = text.split(" until ")
        # sometimes we use "or until" instead of "until", so remove "or"
        text = text.replace(" or", "")
    else:
        termination = None

    # extract duration based on "for" keyword
    if "for" in text:
        # e.g. "Charge at 4 A for 3 hours"
        text, duration = text.split(" for ")
    else:
        duration = None

    if termination is None and duration is None:
        raise ValueError(
            "Operating conditions must contain keyword 'for' or 'until'. "
            f"For example: {_examples}"
        )

    # read remaining instruction
    if text.startswith("Rest"):
        step_class = Rest
        value = 0
    elif text.startswith("Run"):
        raise ValueError(
            "Simulating drive cycles with 'Run' has been deprecated. Use the "
            "pybamm.step.current/voltage/power/c_rate/resistance() functions "
            "instead."
        )
    else:
        # split by what is before and after "at"
        # e.g. "Charge at 4 A" -> ["Charge", "4 A"]
        # e.g. "Discharge at C/2" -> ["Discharge", "C/2"]
        instruction, value_string = text.split(" at ")
        if instruction == "Charge":
            sign = -1
        elif instruction in ["Discharge", "Hold"]:
            sign = 1
        else:
            raise ValueError(
                "Instruction must be 'discharge', 'charge', 'rest', or 'hold'. "
                f"For example: {_examples}"
                f"The following instruction does not comply: {instruction}"
            )
        # extract units (type) and convert value to float
        typ, value = _convert_electric(value_string)
        # Make current positive for discharge and negative for charge
        value *= sign

        # Use the appropriate step class
        step_class = {
            "current": Current,
            "voltage": Voltage,
            "power": Power,
            "C-rate": CRate,
            "resistance": Resistance,
        }[typ]

    return step_class(
        value,
        duration=duration,
        termination=termination,
        description=description,
        **kwargs,
    )


class Current(BaseStepExplicit):
    """
    Current-controlled step, see :class:`pybamm.step.BaseStep` for arguments.
    Current is positive for discharge and negative for charge.
    """

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def current_value(self, variables):
        return self.value


def current(value, **kwargs):
    """
    Current-controlled step, see :class:`pybamm.step.Current`.
    """
    return Current(value, **kwargs)


class Rest(Current):
    """
    Rest step, implemented as a zero-current explicit step.
    """

    def __init__(self, value=0, **kwargs):
        if value != 0:
            raise ValueError("Rest steps must have a current value of 0")
        super().__init__(0, **kwargs)


class CRate(BaseStepExplicit):
    """
    C-rate-controlled step, see :class:`pybamm.step.BaseStep` for arguments.
    C-rate is positive for discharge and negative for charge.
    """

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def current_value(self, variables):
        return self.value * pybamm.Parameter("Nominal cell capacity [A.h]")

    @property
    def _control_target(self):
        # ``value`` is a C-rate; the controlled quantity is the current it sets.
        return self.current_value(None)

    def _default_timespan(self, value):
        # "value" is C-rate, so duration is "1 / value" hours in seconds
        # with a 2x safety factor
        return 1 / abs(value) * 3600 * 2


def c_rate(value, **kwargs):
    """
    C-rate-controlled step, see :class:`pybamm.step.CRate`.
    """
    return CRate(value, **kwargs)


class Voltage(BaseStepImplicit):
    """
    Voltage-controlled step, see :class:`pybamm.step.BaseStep` for arguments.
    Voltage should always be positive.
    """

    control_kind = ControlKind.VOLTAGE

    def get_parameter_values(self, variables):
        return {"Voltage function [V]": self.value}

    def _get_control_residual(self, variables, target):
        return variables["Voltage [V]"] - target

    def get_submodel(self, model):
        return pybamm.external_circuit.VoltageFunctionControl(
            model.param, model.options
        )


def voltage(*args, **kwargs):
    """
    Voltage-controlled step, see :class:`pybamm.step.Voltage`.
    """
    return Voltage(*args, **kwargs)


class Power(BaseStepImplicit):
    """
    Power-controlled step.
    Power is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The value of the power function [W].
    **kwargs
        Any other keyword arguments are passed to the step class
    """

    control_kind = ControlKind.POWER

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def get_parameter_values(self, variables):
        return {"Power function [W]": self.value}

    def _get_control_residual(self, variables, target):
        return variables["Power [W]"] - target

    def get_submodel(self, model):
        return pybamm.external_circuit.PowerFunctionControl(model.param, model.options)


def power(value, **kwargs):
    """
    Power-controlled step, see :class:`pybamm.step.Power`.
    """
    return Power(value, **kwargs)


class Resistance(BaseStepImplicit):
    """
    Resistance-controlled step.
    Resistance is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The value of the power function [W].
    **kwargs
        Any other keyword arguments are passed to the step class
    """

    control_kind = ControlKind.RESISTANCE

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def get_parameter_values(self, variables):
        return {"Resistance function [Ohm]": self.value}

    def _get_control_residual(self, variables, target):
        return variables["Voltage [V]"] - target * variables["Current [A]"]

    def get_submodel(self, model):
        return pybamm.external_circuit.ResistanceFunctionControl(
            model.param, model.options
        )


def resistance(value, **kwargs):
    """
    Resistance-controlled step, see :class:`pybamm.step.Resistance`.
    """
    return Resistance(value, **kwargs)


def rest(duration=None, **kwargs):
    """
    Create a rest step (see :class:`pybamm.step.Rest`).
    """
    return Rest(duration=duration, **kwargs)


class CustomStepExplicit(BaseStepExplicit):
    """
    Custom step class where the current value is explicitly given as a function of
    other variables. When using this class, the user must be careful not to create
    an expression that depends on the current itself, as this will lead to a
    circular dependency. For example, in some models, the voltage is an explicit
    function of the current, so the user should not create a step that depends on
    the voltage. An expression that works for one model may not work for another.

    Parameters
    ----------
    current_value_function : callable
        A function that takes in a dictionary of variables and returns the current
        value.
    duration : float, optional
        The duration of the step in seconds.
    termination : str or list, optional
        A string or list of strings indicating the condition(s) that will terminate the
        step. If a list, the step will terminate when any of the conditions are met.
    period : float or string, optional
        The period of the step. If a float, the value is in seconds. If a string, the
        value should be a valid time string, e.g. "1 hour".
    temperature : float or string, optional
        The temperature of the step. If a float, the value is in Kelvin. If a string,
        the value should be a valid temperature string, e.g. "25 oC".
    tags : str or list, optional
        A string or list of strings indicating the tags associated with the step.
    start_time : str or datetime, optional
        The start time of the step.
    description : str, optional
        A description of the step.
    direction : str, optional
        The direction of the step, e.g. "Charge" or "Discharge" or "Rest".

    Examples
    --------
    Control the current to always be equal to a target power divided by voltage
    (this is one way to implement a power control step):

    >>> def current_function(variables):
    ...     P = 4
    ...     V = variables["Voltage [V]"]
    ...     return P / V

    Create the step with a 2.5 V termination condition:

    >>> step = pybamm.step.CustomStepExplicit(current_function, termination="2.5V")
    """

    def __init__(self, current_value_function, **kwargs):
        super().__init__(None, **kwargs)
        self.current_value_function = current_value_function
        self.kwargs = kwargs

    def current_value(self, variables):
        return self.current_value_function(variables)

    def copy(self):
        return CustomStepExplicit(self.current_value_function, **self.kwargs)


class CustomStepImplicit(BaseStepImplicit):
    """
    Custom step, see :class:`pybamm.step.BaseStep` for arguments.

    Parameters
    ----------
    current_rhs_function : callable
        A function that takes in a dictionary of variables and returns the equation
        controlling the current.

    control : str, optional
        Whether the control is algebraic or differential. Default is algebraic, in
        which case the equation is

        .. math::
            0 = f(\\text{{variables}})

        where :math:`f` is the current_rhs_function.

        If control is "differential", the equation is

        .. math::
            \\frac{dI}{dt} = f(\\text{{variables}})

    duration : float, optional
        The duration of the step in seconds.
    termination : str or list, optional
        A string or list of strings indicating the condition(s) that will terminate the
        step. If a list, the step will terminate when any of the conditions are met.
    period : float or string, optional
        The period of the step. If a float, the value is in seconds. If a string, the
        value should be a valid time string, e.g. "1 hour".
    temperature : float or string, optional
        The temperature of the step. If a float, the value is in Kelvin. If a string,
        the value should be a valid temperature string, e.g. "25 oC".
    tags : str or list, optional
        A string or list of strings indicating the tags associated with the step.
    start_time : str or datetime, optional
        The start time of the step.
    description : str, optional
        A description of the step.
    direction : str, optional
        The direction of the step, e.g. "Charge" or "Discharge" or "Rest".

    Examples
    --------
    Control the current so that the voltage is constant (without using the built-in
    voltage control):

    >>> def voltage_control(variables):
    ...     V = variables["Voltage [V]"]
    ...     return V - 4.2

    Create the step with a duration of 1h. In this case we don't need to specify that
    the control is algebraic, as this is the default.

    >>> step = pybamm.step.CustomStepImplicit(voltage_control, duration=3600)

    Alternatively, control the current by a differential equation to achieve a
    target power:

    >>> def power_control(variables):
    ...     V = variables["Voltage [V]"]
    ...     # Large time constant to avoid large overshoot. The user should be careful
    ...     # to choose a time constant that is appropriate for the model being used,
    ...     # as well as choosing the appropriate sign for the time constant.
    ...     K_V = 100
    ...     return K_V * (V - 4.2)

    Create the step with a 2.5 V termination condition. Now we need to specify that
    the control is differential.

    >>> step = pybamm.step.CustomStepImplicit(
    ...     power_control, termination="2.5V", control="differential"
    ... )
    """

    def __init__(self, current_rhs_function, control="algebraic", **kwargs):
        super().__init__(None, **kwargs)
        self.current_rhs_function = current_rhs_function
        if control not in ["algebraic", "differential"]:
            raise ValueError("control must be either 'algebraic' or 'differential'")
        self.control = control
        self.kwargs = kwargs

    def _get_control_residual(self, variables, target):
        # Custom control is never collapsible, so ``target`` is unused -- the residual is
        # fully defined by ``current_rhs_function``.
        if self.control != "algebraic":
            raise NotImplementedError(
                "Unified experiment control only supports algebraic implicit custom steps"
            )
        return self.current_rhs_function(variables)

    def get_submodel(self, model):
        return pybamm.external_circuit.FunctionControl(
            model.param, self.current_rhs_function, model.options, control=self.control
        )

    def copy(self):
        return CustomStepImplicit(
            self.current_rhs_function, self.control, **self.kwargs
        )


class ControlStep(BaseStep):
    """
    Step that drives model parameters, its controls. It sets each control to a value
    (explicit), or adjusts it so that a residual is zero (implicit). Controls it does
    not drive keep their parameter values.

    Values and targets can be a number, a 2-column array of times and values (as a
    drive cycle), or a function of the time since the start of the step. Steps of the
    same structure whose values and targets are numbers share one branch of the
    unified experiment model.

    Subclasses can override :meth:`control_residuals` to define their own control laws.

    Parameters
    ----------
    explicit : dict, optional
        Maps the name of each control that the step sets to its value.
    implicit : dict, optional
        Maps the name of each control that the step adjusts to its residual: either a
        function of the model variables, or a tuple ``(function, target)`` in which
        ``function`` takes the variables and the target.
    **kwargs
        Any other keyword arguments are passed to :class:`pybamm.step.BaseStep`.

    Examples
    --------
    Set the heating power and the coolant temperature:

    >>> step = pybamm.step.ControlStep(
    ...     explicit={"Heating power [W]": 100, "Coolant temperature [K]": 290},
    ...     duration=3600,
    ... )

    Hold the coolant temperature, and adjust the heating power to keep the net power
    at 50 W:

    >>> def net_power_residual(variables, target):
    ...     return variables["Net power [W]"] - target
    >>> step = pybamm.step.ControlStep(
    ...     explicit={"Coolant temperature [K]": 290},
    ...     implicit={"Heating power [W]": (net_power_residual, 50)},
    ...     duration=3600,
    ... )
    """

    def __init__(self, explicit=None, implicit=None, **kwargs):
        explicit = dict(explicit or {})
        implicit = dict(implicit or {})
        if not explicit and not implicit:
            raise ValueError("A control step must set or adjust at least one control")
        both = sorted(explicit.keys() & implicit.keys())
        if both:
            raise ValueError(f"Controls {both} cannot be both set and adjusted")
        self.residual_functions = {}
        input_targets = dict(explicit)
        for name, residual in implicit.items():
            if isinstance(residual, tuple):
                self.residual_functions[name], input_targets[name] = residual
            else:
                self.residual_functions[name] = residual
            if not callable(self.residual_functions[name]):
                raise TypeError(f"The residual of '{name}' must be a function")
        for value in input_targets.values():
            if isinstance(value, np.ndarray):
                _check_drive_cycle(value)
            elif callable(value):
                _check_function_of_time(value)
        self.input_explicit = explicit
        self.input_implicit = implicit
        self.input_targets = input_targets
        super().__init__(None, **kwargs)
        self.controls = (*explicit, *implicit)
        #: The step's own target of each control, as a number or a symbol, keyed by
        #: control name. Implicit controls without a target are left out.
        self.targets = {
            name: self._process_target(value) for name, value in input_targets.items()
        }
        self.control_kind = self._control_structure()

    def _process_target(self, value):
        if isinstance(value, np.ndarray):
            return _drive_cycle_interpolant(value, self.duration)
        if callable(value):
            return value(_step_time())
        return value

    def _control_structure(self):
        """Everything that defines the control law of the step except its targets."""
        return (
            f"{type(self).__name__}(explicit={list(self.input_explicit)}, "
            f"implicit={self.residual_functions})"
        )

    def _has_drive_cycle(self):
        return any(
            isinstance(value, np.ndarray) for value in self.input_targets.values()
        )

    def default_duration(self, value):
        ends = [
            value[-1, 0]
            for value in self.input_targets.values()
            if isinstance(value, np.ndarray)
        ]
        return max(ends) if ends else self._default_timespan(value)

    def record_tags(self, value, *args, **kwargs):
        repr_args, _ = super().record_tags(self._controls_text(repr), *args, **kwargs)
        _, hash_args = super().record_tags(
            self._controls_text(_value_identity), *args, **kwargs
        )
        return repr_args, hash_args

    def _controls_text(self, value_text):
        """The explicit and implicit controls as text, with values as ``value_text``."""

        def entry_text(entry):
            if isinstance(entry, tuple):
                return f"({', '.join(value_text(item) for item in entry)})"
            return value_text(entry)

        parts = []
        for label, controls in (
            ("explicit", self.input_explicit),
            ("implicit", self.input_implicit),
        ):
            if controls:
                entries = ", ".join(
                    f"{name!r}: {entry_text(entry)}" for name, entry in controls.items()
                )
                parts.append(f"{label}={{{entries}}}")
        return ", ".join(parts)

    def copy(self):
        # A shallow copy keeps the processed targets and supports subclasses with
        # other arguments
        return copy.copy(self)

    def is_implicit(self):
        return bool(self.input_implicit)

    def to_dict(self):
        step_dict = super().to_dict()
        del step_dict["value"]
        step_dict["explicit"] = self.input_explicit
        step_dict["implicit"] = self.input_implicit
        return step_dict

    def control_target_values(self, parameter_values):
        return {
            name: _constant_value(parameter_values, self.targets.get(name))
            for name in self.controls
        }

    def get_control_residuals(self, variables, targets=None):
        targets = {**self.targets, **(targets or {})}
        return self.control_residuals(variables, targets)

    def control_residuals(self, variables, targets):
        """
        Residual of each control, which the control law makes zero.

        Parameters
        ----------
        variables : dict
            The model variables, including the controls.
        targets : dict
            The target of each control, as a symbol, keyed by control name.

        Returns
        -------
        dict
            The residual of each control in :attr:`controls`, keyed by control name.
        """
        residuals = {
            name: _control_variable(variables, name) - targets[name]
            for name in self.input_explicit
        }
        for name, function in self.residual_functions.items():
            if isinstance(self.input_implicit[name], tuple):
                residuals[name] = function(variables, targets[name])
            else:
                residuals[name] = function(variables)
        return residuals

    def set_up(self, new_model, new_parameter_values):
        for name in self.input_explicit:
            new_parameter_values[name] = self.targets[name]

        variables = new_model.variables
        control_variables = {}
        for name in self.input_implicit:
            default = _default_control_value(new_model, new_parameter_values, name)
            control_variables[name] = _adjusted_control_variable(name, default)
            new_model.initial_conditions[control_variables[name]] = (
                _initial_control_guess(default)
            )
            variables[name] = control_variables[name]
        residuals = self.get_control_residuals(variables)
        for name, variable in control_variables.items():
            new_model.algebraic[variable] = residuals[name]
            new_parameter_values[name] = variable
        return new_model, new_parameter_values


def control(explicit=None, implicit=None, **kwargs):
    """
    Step that drives any of the model's parameters, see
    :class:`pybamm.step.ControlStep`.
    """
    return ControlStep(explicit, implicit, **kwargs)
