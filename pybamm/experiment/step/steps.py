#
# Public functions to create steps for use in an experiment.
#
import pybamm
from .base_step import (
    BaseStepExplicit,
    BaseStepAlgebraic,
    _convert_electric,
    _examples,
)


def string(string, **kwargs):
    """
    Create a step from a string.

    Parameters
    ----------
    string : str
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
    if not isinstance(string, str):
        raise TypeError("Input to step.string() must be a string")

    if "oC" in string:
        raise ValueError(
            "Temperature must be specified as a keyword argument "
            "instead of in the string"
        )

    # Save the original string
    description = string

    # extract period
    if "period)" in string:
        if "period" in kwargs:
            raise ValueError(
                "Period cannot be specified both as a keyword argument "
                "and in the string"
            )
        string, period_full = string.split(" (")
        period, _ = period_full.split(" period)")
        kwargs["period"] = period
    # extract termination condition based on "until" keyword
    if "until" in string:
        # e.g. "Charge at 4 A until 3.8 V"
        string, termination = string.split(" until ")
        # sometimes we use "or until" instead of "until", so remove "or"
        string = string.replace(" or", "")
    else:
        termination = None

    # extract duration based on "for" keyword
    if "for" in string:
        # e.g. "Charge at 4 A for 3 hours"
        string, duration = string.split(" for ")
    else:
        duration = None

    if termination is None and duration is None:
        raise ValueError(
            "Operating conditions must contain keyword 'for' or 'until'. "
            f"For example: {_examples}"
        )

    # read remaining instruction
    if string.startswith("Rest"):
        step_class = Current
        value = 0
    elif string.startswith("Run"):
        raise ValueError(
            "Simulating drive cycles with 'Run' has been deprecated. Use the "
            "pybamm.step.current/voltage/power/c_rate/resistance() functions "
            "instead."
        )
    else:
        # split by what is before and after "at"
        # e.g. "Charge at 4 A" -> ["Charge", "4 A"]
        # e.g. "Discharge at C/2" -> ["Discharge", "C/2"]
        instruction, value_string = string.split(" at ")
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

    def current_value(self, variables):
        return self.value * pybamm.Parameter("Nominal cell capacity [A.h]")


def c_rate(value, **kwargs):
    """
    C-rate-controlled step, see :class:`pybamm.step.CRate`.
    """
    return CRate(value, **kwargs)


class Voltage(BaseStepAlgebraic):
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


class Power(BaseStepAlgebraic):
    """
    Power-controlled step.
    Power is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The value of the power function [W].
    control : str, optional
        Whether the control is algebraic or differential. Default is algebraic.
    **kwargs
        Any other keyword arguments are passed to the step class
    """

    def __init__(self, value, control="algebraic", **kwargs):
        super().__init__(value, **kwargs)
        self.control = control

    def get_parameter_values(self, variables):
        return {"Power function [W]": self.value}

    def get_submodel(self, model):
        return pybamm.external_circuit.PowerFunctionControl(
            model.param, model.options, control=self.control
        )


def power(value, **kwargs):
    """
    Power-controlled step, see :class:`pybamm.step.Power`.
    """
    return Power(value, **kwargs)


class Resistance(BaseStepAlgebraic):
    """
    Resistance-controlled step.
    Resistance is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The value of the power function [W].
    control : str, optional
        Whether the control is algebraic or differential. Default is algebraic.
    **kwargs
        Any other keyword arguments are passed to the step class
    """

    def __init__(self, value, control="algebraic", **kwargs):
        super().__init__(value, **kwargs)
        self.control = control

    def get_parameter_values(self, variables):
        return {"Resistance function [Ohm]": self.value}

    def get_submodel(self, model):
        return pybamm.external_circuit.ResistanceFunctionControl(
            model.param, model.options, control=self.control
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
