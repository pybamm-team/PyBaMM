#
# Classes for experimental steps
#

examples = """

    "Discharge at 1C for 0.5 hours",
    "Discharge at C/20 for 0.5 hours",
    "Charge at 0.5 C for 45 minutes",
    "Discharge at 1 A for 0.5 hours",
    "Charge at 200 mA for 45 minutes",
    "Discharge at 1W for 0.5 hours",
    "Charge at 200mW for 45 minutes",
    "Rest for 10 minutes",
    "Hold at 1V for 20 seconds",
    "Charge at 1 C until 4.1V",
    "Hold at 4.1 V until 50mA",
    "Hold at 3V until C/50",
    "Discharge at C/3 for 2 hours or until 2.5 V",

    """


class _Step:
    """
    Class representing one step in an experiment.
    All experiment steps are functions that return an instance of this class.
    This class is not intended to be used directly.

    Parameters
    ----------
    typ : str
        The type of step, can be "current", "voltage", "cccv_ode" or "rest", "power",
        or "resistance".
    value : float
        The value of the step, corresponding to the type of step. Can be a number, a
        2-tuple (for cccv_ode), or a 2-column array (for drive cycles)
    duration : float, optional
        The duration of the step in seconds.

    """

    def __init__(
        self,
        typ,
        value,
        duration=None,
        termination=None,
        period=None,
        temperature=None,
        tags=None,
    ):
        self.typ = typ
        self.value = value
        self.duration = _convert_time_to_seconds(duration)

        if termination is not None:
            termination = _convert_electric(termination)
            self.termination = {"type": termination[0], "value": termination[1]}
        else:
            self.termination = None

        self.period = _convert_time_to_seconds(period)
        self.temperature = _convert_temperature_to_kelvin(temperature)

        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        self.tags = tags

    def __repr__(self):
        return f"Step({self.typ}, {self.value}, {self.duration})"

    def to_dict(self):
        """
        Convert the step to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the step information.
        """
        return {
            "type": self.typ,
            "value": self.value,
            "duration": self.duration,
            "termination": self.termination,
            "period": self.period,
            "temperature": self.temperature,
            "tags": self.tags,
        }


def _convert_time_to_seconds(time_and_units):
    """Convert a time in seconds, minutes or hours to a time in seconds"""
    # If the time is a number, assume it is in seconds
    if isinstance(time_and_units, (int, float)) or time_and_units is None:
        return time_and_units

    # Split number and units
    units = time_and_units.lstrip("0123456789.- ")
    time = time_and_units[: -len(units)]
    if units in ["second", "seconds", "s", "sec"]:
        time_in_seconds = float(time)
    elif units in ["minute", "minutes", "m", "min"]:
        time_in_seconds = float(time) * 60
    elif units in ["hour", "hours", "h", "hr"]:
        time_in_seconds = float(time) * 3600
    else:
        raise ValueError(
            "time units must be 'seconds', 'minutes' or 'hours'. "
            f"For example: {examples}"
        )
    return time_in_seconds


def _convert_temperature_to_kelvin(temperature_and_units):
    """Convert a temperature in Celsius or Kelvin to a temperature in Kelvin"""
    # If the temperature is a number, assume it is in Kelvin
    if isinstance(temperature_and_units, (int, float)) or temperature_and_units is None:
        return temperature_and_units

    # Split number and units
    units = temperature_and_units.lstrip("0123456789. ")
    temperature = temperature_and_units[: -len(units)]
    if units in ["K"]:
        temperature_in_kelvin = float(temperature)
    elif units in ["oC"]:
        temperature_in_kelvin = float(temperature) + 273.15
    else:
        raise ValueError("temperature units must be 'K' or 'oC'. ")
    return temperature_in_kelvin


def _convert_electric(value_string):
    """Convert electrical instructions to consistent output"""
    if value_string is None:
        return None
    # Special case for C-rate e.g. C/2
    if value_string[0] == "C":
        unit = "C"
        value = 1 / float(value_string[2:])
    else:
        # All other cases e.g. 4 A, 2.5 V, 1.5 Ohm
        unit = value_string.lstrip("0123456789.- ")
        value = float(value_string[: -len(unit)])
        # Catch milli- prefix
        if unit.startswith("m"):
            unit = unit[1:]
            value /= 1000

    # Convert units to type
    units_to_type = {
        "C": "C-rate",
        "A": "current",
        "V": "voltage",
        "W": "power",
        "Ohm": "resistance",
    }
    try:
        typ = units_to_type[unit]
    except KeyError:
        raise ValueError(
            f"units must be 'A', 'V', 'W', 'Ohm', or 'C'. For example: {examples}"
        )
    return typ, value


def _read_instruction_value(instruction_value):
    if instruction_value.startswith("Rest"):
        return ("current", 0)
    else:
        # split by what is before and after "at"
        # e.g. "Charge at 4 A" -> ["Charge", "4 A"]
        # e.g. "Discharge at C/2" -> ["Discharge", "C/2"]
        instruction, value_string = instruction_value.split(" at ")
        if instruction == "Charge":
            sign = -1
        elif instruction in ["Discharge", "Hold"]:
            sign = 1
        else:
            raise ValueError(
                "Instruction must be 'discharge', 'charge', 'rest', or 'hold'. "
                f"For example: {examples}"
                f"The following instruction does not comply: {instruction}"
            )
        # extract units (type) and convert value to float
        typ, value = _convert_electric(value_string)
        return typ, sign * value


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
    # extract termination condition based on "until" keyword
    if "until" in string:
        # e.g. "Charge at 4 A until 3.8 V"
        string, termination = string.split(" until ")
        # sometimes we use "or until" instead of "until", so remove "or"
        string.replace(" or", "")
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
            f"For example: {examples}"
        )

    # read remaining instruction
    typ, value = _read_instruction_value(string)

    return _Step(typ, value, duration=duration, termination=termination, **kwargs)


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


def rest(**kwargs):
    """
    Create a rest step, equivalent to a constant current step with value 0
    (see :meth:`pybamm.experiment.current`).
    """
    return current(0, **kwargs)


def cccv_ode(current, voltage, **kwargs):
    """
    Create a constant current constant voltage step, to be solved in one go using an
    ODE for the current. This is different from a constant current step followed by a
    constant voltage step, which is solved in two steps.

    Parameters
    ----------
    current : float
        The current value in A.
    voltage : float
        The voltage value in V.
    **kwargs
        Any other keyword arguments are passed to the :class:`pybamm.experiment.Step`
        class.

    Returns
    -------
    :class:`pybamm.experiment.Step`
        A constant current constant voltage step.
    """
    return _Step("cccv_ode", (current, voltage), **kwargs)
