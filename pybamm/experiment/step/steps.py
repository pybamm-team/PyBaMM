import pybamm
from .base_step import (
    BaseStepExplicit,
    BaseStepImplicit,
    _convert_electric,
    _examples,
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
        step_class = Current
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

    def default_duration(self, value):
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

    def get_parameter_values(self, variables):
        return {"Voltage function [V]": self.value}

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

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def get_parameter_values(self, variables):
        return {"Power function [W]": self.value}

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

    def __init__(self, value, **kwargs):
        self.calculate_charge_or_discharge = True
        super().__init__(value, **kwargs)

    def get_parameter_values(self, variables):
        return {"Resistance function [Ohm]": self.value}

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
    Create a rest step, equivalent to a constant current step with value 0
    (see :meth:`pybamm.step.current`).
    """
    return Current(0, duration=duration, **kwargs)


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

    def get_submodel(self, model):
        return pybamm.external_circuit.FunctionControl(
            model.param, self.current_rhs_function, model.options, control=self.control
        )

    def copy(self):
        return CustomStepImplicit(
            self.current_rhs_function, self.control, **self.kwargs
        )
