#
# Private classes and functions for experiment steps
#
import pybamm
import numpy as np
from datetime import datetime
from .step_termination import _read_termination
import numbers

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


class BaseStep:
    """
    Class representing one step in an experiment.
    All experiment steps are functions that return an instance of this class.
    This class is not intended to be used directly, but can be subtyped to create a
    custom experiment step.

    Parameters
    ----------
    value : float
        The value of the step, corresponding to the type of step. Can be a number, a
        2-tuple (for cccv_ode), a 2-column array (for drive cycles), or a 1-argument function of t
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
    """

    def __init__(
        self,
        value,
        duration=None,
        termination=None,
        period=None,
        temperature=None,
        tags=None,
        start_time=None,
        description=None,
        direction=None,
    ):
        self.input_duration = duration
        self.input_value = value
        # Check if drive cycle
        is_drive_cycle = isinstance(value, np.ndarray)
        is_python_function = callable(value)
        if is_drive_cycle:
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError(
                    "Drive cycle must be a 2-column array with time in the first column"
                    " and current/C-rate/power/voltage/resistance in the second"
                )
            # Check that drive cycle starts at t=0
            t = value[:, 0]
            if t[0] != 0:
                raise ValueError("Drive cycle must start at t=0")
        elif is_python_function:
            t0 = 0
            # Check if the function is only a function of t
            try:
                value_t0 = value(t0)
            except TypeError:
                raise TypeError(
                    "Input function must have only 1 positional argument for time"
                ) from None

            # Check if the value at t0 is feasible
            if not (np.isfinite(value_t0) and np.isscalar(value_t0)):
                raise ValueError(
                    f"Input function must return a real number output at t = {t0}"
                )

        # Record whether the step uses the default duration
        # This will be used by the experiment to check whether the step is feasible
        self.uses_default_duration = duration is None
        # Set duration
        if self.uses_default_duration:
            duration = self.default_duration(value)
        self.duration = _convert_time_to_seconds(duration)

        # If drive cycle, repeat the drive cycle until the end of the experiment,
        # and create an interpolant
        if is_drive_cycle:
            t_max = self.duration
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

            self.value = pybamm.Interpolant(
                t,
                y,
                pybamm.t - pybamm.InputParameter("start time"),
                name="Drive Cycle",
            )
            self.period = np.diff(t).min()
        elif is_python_function:
            t = pybamm.t - pybamm.InputParameter("start time")
            self.value = value(t)
            self.period = _convert_time_to_seconds(period)
        else:
            self.value = value
            self.period = _convert_time_to_seconds(period)

        if (
            hasattr(self, "calculate_charge_or_discharge")
            and self.calculate_charge_or_discharge
        ):
            direction = self.value_based_charge_or_discharge()
        self.direction = direction

        self.repr_args, self.hash_args = self.record_tags(
            value,
            duration,
            termination,
            period,
            temperature,
            tags,
            start_time,
            description,
            direction,
        )

        self.description = description

        if termination is None:
            termination = []
        elif not isinstance(termination, list):
            termination = [termination]
        self.termination = []
        for term in termination:
            if isinstance(term, str):
                term = _convert_electric(term)
            term = _read_termination(term)
            self.termination.append(term)

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

    def copy(self):
        """
        Return a copy of the step.

        Returns
        -------
        :class:`pybamm.Step`
            A copy of the step.
        """
        return self.__class__(
            self.input_value,
            duration=self.input_duration,
            termination=self.termination,
            period=self.period,
            temperature=self.temperature,
            tags=self.tags,
            start_time=self.start_time,
            description=self.description,
            direction=self.direction,
        )

    def __str__(self):
        if self.description is not None:
            return self.description
        else:
            return repr(self)

    def __repr__(self):
        return f"Step({self.repr_args})"

    def basic_repr(self):
        """
        Return a basic representation of the step, only with type, value, termination
        and temperature, which are the variables involved in processing the model. Also
        used for hashing.
        """
        return f"Step({self.hash_args})"

    def to_dict(self):
        """
        Convert the step to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the step information.
        """
        return {
            "type": self.__class__.__name__,
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
        return isinstance(other, BaseStep) and self.hash_args == other.hash_args

    def __hash__(self):
        return hash(self.basic_repr())

    def default_duration(self, value):
        """
        Default duration for the step is one day (24 hours) or the duration of the
        drive cycle
        """
        if isinstance(value, np.ndarray):
            t = value[:, 0]
            return t[-1]
        else:
            return 24 * 3600  # one day in seconds

    def process_model(self, model, parameter_values):
        new_model = model.new_copy()
        new_parameter_values = parameter_values.copy()
        new_model, new_parameter_values = self.set_up(new_model, new_parameter_values)
        self.update_model_events(new_model)

        # Update temperature
        if self.temperature is not None:
            new_parameter_values["Ambient temperature [K]"] = self.temperature

        # Parameterise the model
        parameterised_model = new_parameter_values.process_model(
            new_model, inplace=False
        )

        return parameterised_model

    def update_model_events(self, new_model):
        for term in self.termination:
            event = term.get_event(new_model.variables, self)
            if event is not None:
                new_model.events.append(event)

        # Keep the min and max voltages as safeguards but add some tolerances
        # so that they are not triggered before the voltage limits in the
        # experiment
        for i, event in enumerate(new_model.events):
            if event.name in ["Minimum voltage [V]", "Maximum voltage [V]"]:
                new_model.events[i] = pybamm.Event(
                    event.name, event.expression + 1, event.event_type
                )

    def value_based_charge_or_discharge(self):
        """
        Determine whether the step is a charge or discharge step based on the value of the
        step
        """
        if isinstance(self.value, pybamm.Symbol):
            inpt = {"start time": 0}
            init_curr = self.value.evaluate(t=0, inputs=inpt).flatten()[0]
        else:
            init_curr = self.value
        sign = np.sign(init_curr)
        if sign == 0:
            return "Rest"
        elif sign > 0:
            return "Discharge"
        else:
            return "Charge"

    def record_tags(
        self,
        value,
        duration,
        termination,
        period,
        temperature,
        tags,
        start_time,
        description,
        direction,
    ):
        """Record all the args for repr and hash"""
        repr_args = f"{value}, duration={duration}"
        hash_args = f"{value}"
        if termination:
            repr_args += f", termination={termination}"
            hash_args += f", termination={termination}"
        if period:
            repr_args += f", period={period}"
        if temperature:
            repr_args += f", temperature={temperature}"
            hash_args += f", temperature={temperature}"
        if tags:
            repr_args += f", tags={tags}"
        if start_time:
            repr_args += f", start_time={start_time}"
        if description:
            repr_args += f", description={description}"
        if direction:
            repr_args += f", direction={direction}"
            hash_args += f", direction={direction}"
        return repr_args, hash_args


class BaseStepExplicit(BaseStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def current_value(self, variables):
        raise NotImplementedError

    def set_up(self, new_model, new_parameter_values):
        new_parameter_values["Current function [A]"] = self.current_value(
            new_model.variables
        )
        return new_model, new_parameter_values


class BaseStepImplicit(BaseStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_parameter_values(self, variables):
        return {}

    def get_submodel(self, model):
        raise NotImplementedError

    def set_up(self, new_model, new_parameter_values):
        # Create a new model where the current density is now a variable
        # To do so, we replace all instances of the current density in the
        # model with a current density variable, which is obtained from the
        # FunctionControl submodel
        # check which kind of external circuit model we need (differential
        # or algebraic)
        # Build the new submodel and update the model with it
        submodel = self.get_submodel(new_model)
        variables = new_model.variables
        submodel.variables = submodel.get_fundamental_variables()
        variables.update(submodel.variables)
        submodel.variables.update(submodel.get_coupled_variables(variables))
        variables.update(submodel.variables)
        submodel.set_rhs(variables)
        submodel.set_algebraic(variables)
        submodel.set_initial_conditions(variables)
        new_model.rhs.update(submodel.rhs)
        new_model.algebraic.update(submodel.algebraic)
        new_model.initial_conditions.update(submodel.initial_conditions)

        # Set the "current function" to be the variable defined in the submodel
        new_parameter_values["Current function [A]"] = submodel.variables["Current [A]"]
        # Update any other parameters as necessary
        new_parameter_values.update(
            self.get_parameter_values(variables), check_already_exists=False
        )

        return new_model, new_parameter_values


_type_to_units = {
    "current": "[A]",
    "voltage": "[V]",
    "power": "[W]",
    "resistance": "[Ohm]",
}


def _convert_time_to_seconds(time_and_units):
    """Convert a time in seconds, minutes or hours to a time in seconds"""
    if time_and_units is None:
        return time_and_units

    # If the time is a number, assume it is in seconds
    if isinstance(time_and_units, numbers.Number):
        if time_and_units <= 0:
            raise ValueError("time must be positive")
        else:
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
    except KeyError as error:
        raise ValueError(
            f"units must be 'A', 'V', 'W', 'Ohm', or 'C'. For example: {_examples}"
        ) from error
    return typ, value
