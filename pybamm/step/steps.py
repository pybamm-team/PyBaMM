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
        The string to parse. Each operating condition should
        be of the form "Do this for this long" or "Do this until this happens". For
        example, "Charge at 1 C for 1 hour", or "Charge at 1 C until 4.2 V", or "Charge
        at 1 C for 1 hour or until 4.2 V". The instructions can be of the form
        "(Dis)charge at x A/C/W", "Rest", or "Hold at x V until y A". The running
        time should be a time in seconds, minutes or hours, e.g. "10 seconds",
        "3 minutes" or "1 hour". The stopping conditions should be
        a circuit state, e.g. "1 A", "C/50" or "3 V".
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
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
        typ = "current"
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
        The current value in A. It can be a number or a 2-column array (for drive cycles).
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
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
        The C-rate value. It can be a number or a 2-column array (for drive cycles).
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
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
        The voltage value in V. It can be a number or a 2-column array (for drive cycles).
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
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
        The power value in W. It can be a number or a 2-column array (for drive cycles).
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
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
        The resistance value in Ohm. It can be a number or a 2-column array (for drive cycles).
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.step._Step`
        class.

    Returns
    -------
    :class:`pybamm.step._Step`
        A resistance-controlled step.
    """
    return _Step("resistance", value, **kwargs)


def rest(duration=None, **kwargs):
    """
    Create a rest step, equivalent to a constant current step with value 0
    (see :meth:`pybamm.step.current`).
    """
    return current(0, duration=duration, **kwargs)
