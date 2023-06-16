#
# Private classes and functions for experiment steps
#
import pybamm
import numpy as np
from datetime import datetime

_examples = """

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
        The type of step, can be "current", "voltage", "c_rate", "power",
        or "resistance".
    value : float
        The value of the step, corresponding to the type of step. Can be a number, a
        2-tuple (for cccv_ode), or a 2-column array (for drive cycles)
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
    datetime : str or datetime, optional
        A string or list of strings indicating the tags associated with the step.
    description : str, optional
        A description of the step.
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
        start_time=None,
        description=None,
    ):
        self.type = typ

        # Record all the args for repr
        self.args = f"{typ}, {value}"
        if duration:
            self.args += f", duration={duration}"
        if termination:
            self.args += f", termination={termination}"
        if period:
            self.args += f", period={period}"
        if temperature:
            self.args += f", temperature={temperature}"
        if tags:
            self.args += f", tags={tags}"
        if start_time:
            self.args += f", start_time={start_time}"
        if description:
            self.args += f", description={description}"

        # Check if drive cycle
        self.is_drive_cycle = isinstance(value, np.ndarray)
        if self.is_drive_cycle:
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError(
                    "Drive cycle must be a 2-column array with time in the first column"
                    " and current/C-rate/power/voltage/resistance in the second"
                )
            if duration is not None:
                t_max = _convert_time_to_seconds(duration)
                self.duration = _convert_time_to_seconds(duration)
                if t_max > value[-1, 0]:
                    # duration longer than drive cycle values so loop
                    nloop = np.ceil(t_max / value[-1, 0]).astype(int)
                    tstep = np.diff(value[:, 0])[0]
                    t = []
                    y = []
                    for i in range(nloop):
                        t.append(value[:, 0] + ((value[-1, 0] + tstep) * i))
                        y.append(value[:, 1])
                    t = np.asarray(t).flatten()
                    y = np.asarray(y).flatten()
                else:
                    t, y = value[:, 0], value[:, 1]
            else:
                t, y = value[:, 0], value[:, 1]
                self.duration = t.max()

            self.value = pybamm.Interpolant(
                t, y, pybamm.t - pybamm.InputParameter("start time")
            )
            self.period = np.diff(t).min()
        else:
            self.value = value
            self.duration = _convert_time_to_seconds(duration)
            self.period = _convert_time_to_seconds(period)

        self.description = description

        if termination is None:
            termination = []
        elif not isinstance(termination, list):
            termination = [termination]
        self.termination = []
        for term in termination:
            typ, value = _convert_electric(term)
            self.termination.append({"type": typ, "value": value})

        self.temperature = _convert_temperature_to_kelvin(temperature)

        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        self.tags = tags

        if start_time is None or isinstance(start_time, datetime):
            self.start_time = start_time
        else:
            raise TypeError("`start_time` should be a datetime.datetime object")
        self.next_start_time = None
        self.end_time = None

    def __str__(self):
        if self.description is not None:
            return self.description
        else:
            return repr(self)

    def __repr__(self):
        return f"_Step({self.args})"

    def to_dict(self):
        """
        Convert the step to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the step information.
        """
        return {
            "type": self.type,
            "value": self.value,
            "duration": self.duration,
            "termination": self.termination,
            "period": self.period,
            "temperature": self.temperature,
            "tags": self.tags,
            "start_time": self.start_time,
            "description": self.description,
        }

    def __eq__(self, other):
        return (
            isinstance(other, _Step)
            and self.__repr__() == other.__repr__()
            and self.next_start_time == other.next_start_time
            and self.end_time == other.end_time
        )

    def __hash__(self):
        return hash(repr(self))

    @property
    def unit(self):
        return _type_to_units[self.type]


_type_to_units = {
    "current": "[A]",
    "voltage": "[V]",
    "power": "[W]",
    "resistance": "[Ohm]",
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
            f"For example: {_examples}"
        )
    return time_in_seconds


def _convert_temperature_to_kelvin(temperature_and_units):
    """Convert a temperature in Celsius or Kelvin to a temperature in Kelvin"""
    # If the temperature is a number, assume it is in Kelvin
    if isinstance(temperature_and_units, (int, float)) or temperature_and_units is None:
        return temperature_and_units

    # Split number and units
    units = temperature_and_units.lstrip("0123456789.- ")
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
            f"units must be 'A', 'V', 'W', 'Ohm', or 'C'. For example: {_examples}"
        )
    return typ, value
