#
# Public functions to create steps for use in an experiment.
#
from ._steps_util import _Step, _convert_electric, _examples


def string(string, **kwargs):
    """
    Create a step from a string.

    Parameters
    ----------
    string : str
        The string to parse. The string must be in the format
        "type:value:duration", where type is either "current" or "voltage", value is
        the value of the step (in A or V) and duration is the duration of the step
        (in seconds). For example, "current:1:3600" is a constant current step of 1 A
        for 1 hour.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A step parsed from the string.
    """
    if not isinstance(string, str):
        raise TypeError("Input to experiment.string() must be a string")

    if "period)" in string:
        raise ValueError(
            "Period must be specified as a keyword argument instead of in the string"
        )

    if "oC" in string:
        raise ValueError(
            "Temperature must be specified as a keyword argument "
            "instead of in the string"
        )

    # Save the original string
    description = string

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
        typ = "current"
        value = 0
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

    return _Step(
        typ,
        value,
        duration=duration,
        termination=termination,
        description=description,
        **kwargs,
    )


def current(value, **kwargs):
    """
    Create a current-controlled step.
    Current is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The current value in A.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A current-controlled step.
    """
    return _Step("current", value, **kwargs)


def c_rate(value, **kwargs):
    """
    Create a C-rate controlled step.
    C-rate is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The C-rate value.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A C-rate controlled step.
    """
    return _Step("C-rate", value, **kwargs)


def voltage(value, **kwargs):
    """
    Create a voltage-controlled step.
    Voltage should always be positive.

    Parameters
    ----------
    value : float
        The voltage value in V.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A voltage-controlled step.
    """
    return _Step("voltage", value, **kwargs)


def power(value, **kwargs):
    """
    Create a power-controlled step.
    Power is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The power value in W.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A power-controlled step.
    """
    return _Step("power", value, **kwargs)


def resistance(value, **kwargs):
    """
    Create a resistance-controlled step.
    Resistance is positive for discharge and negative for charge.

    Parameters
    ----------
    value : float
        The resistance value in Ohm.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A resistance-controlled step.
    """
    return _Step("resistance", value, **kwargs)


def rest(duration=None, **kwargs):
    """
    Create a rest step, equivalent to a constant current step with value 0
    (see :meth:`pybamm.experiment.current`).
    """
    return current(0, duration=duration, **kwargs)
