from __future__ import annotations
import pybamm
from .step.base_step import (
    _convert_time_to_seconds,
    _convert_temperature_to_kelvin,
)


class Experiment:
    """
    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be either a `pybamm.step.BaseStep` class, which can be created using one of the
    methods `pybamm.step.current`, `pybamm.step.c_rate`, `pybamm.step.voltage`
    , `pybamm.step.power`, `pybamm.step.resistance`, or
    `pybamm.step.string`, or a string, in which case the string is passed to
    `pybamm.step.string`.

    Parameters
    ----------
    operating_conditions : list[str]
        List of strings representing the operating conditions.
    period : str, optional
        Period (1/frequency) at which to record outputs. Default is 1 minute. Can be
        overwritten by individual operating conditions.
    temperature: float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    termination : list[str], optional
        List of strings representing the conditions to terminate the experiment. Default is None.
        This is different from the termination for individual steps. Termination for
        individual steps is specified in the step itself, and the simulation moves to
        the next step when the termination condition is met
        (e.g. 2.5V discharge cut-off). Termination for the
        experiment as a whole is specified here, and the simulation stops when the
        termination condition is met (e.g. 80% capacity).
    """

    def __init__(
        self,
        operating_conditions: list[str | tuple[str]],
        period: str = "1 minute",
        temperature: float | None = None,
        termination: list[str] | None = None,
    ):
        # Save arguments for copying
        self.args = (
            operating_conditions,
            period,
            temperature,
            termination,
        )

        cycles = []
        for cycle in operating_conditions:
            if not isinstance(cycle, tuple):
                cycle = (cycle,)
            cycles.append(cycle)
        self.cycles = cycles
        self.cycle_lengths = [len(cycle) for cycle in cycles]

        steps_unprocessed = [cond for cycle in cycles for cond in cycle]

        # Convert strings to pybamm.step.BaseStep objects
        # We only do this once per unique step, to avoid unnecessary conversions
        # Assign experiment period and temperature if not specified in step
        self.period = _convert_time_to_seconds(period)
        self.temperature = _convert_temperature_to_kelvin(temperature)

        processed_steps = self.process_steps(
            steps_unprocessed, self.period, self.temperature
        )

        self.steps = [processed_steps[repr(step)] for step in steps_unprocessed]
        self.steps = self._set_next_start_time(self.steps)

        # Save the processed unique steps and the processed operating conditions
        # for every step
        self.unique_steps = set(processed_steps.values())

        # Allocate experiment global variables
        self.initial_start_time = self.steps[0].start_time

        if self.steps[0].end_time is not None and self.initial_start_time is None:
            raise ValueError(
                "When using experiments with `start_time`, the first step must have a "
                "`start_time`."
            )

        self.termination_string = termination
        self.termination = self.read_termination(termination)

    @staticmethod
    def process_steps(unprocessed_steps, period, temp):
        processed_steps = {}
        for step in unprocessed_steps:
            if repr(step) in processed_steps:
                continue
            elif isinstance(step, str):
                processed_step = pybamm.step.string(step)
            elif isinstance(step, pybamm.step.BaseStep):
                # Copy the step to avoid modifying the original with the period and
                # temperature and any other changes
                processed_step = step.copy()
            else:
                raise TypeError("Operating conditions must be a Step object or string.")

            if processed_step.period is None:
                processed_step.period = period
            if processed_step.temperature is None:
                processed_step.temperature = temp

            processed_steps[repr(step)] = processed_step

        return processed_steps

    def __str__(self):
        return str(self.cycles)

    def copy(self):
        return Experiment(*self.args)

    def __repr__(self):
        return f"pybamm.Experiment({self!s})"

    @staticmethod
    def read_termination(termination):
        """
        Read the termination reason. If this condition is hit, the experiment will stop.

        Parameters
        ----------
        termination : str or list[str], optional
           A single string, or a list of strings, representing the conditions to terminate the experiment.
           Only capacity or voltage can be provided as a termination reason.
           e.g. '4 Ah capacity' or ['80% capacity', '2.5 V']

        Returns
        -------
        dict
           A dictionary of the termination conditions.
           e.g. {'capacity': (4.0, 'Ah')} or
           {'capacity': (80.0, '%'), 'voltage': (2.5, 'V')}

        """
        if termination is None:
            return {}
        elif isinstance(termination, str):
            termination = [termination]

        termination_dict = {}
        for term in termination:
            term_list = term.split()
            if term_list[-1] == "capacity":
                end_discharge = "".join(term_list[:-1])
                end_discharge = end_discharge.replace("A.h", "Ah")
                if end_discharge.endswith("%"):
                    end_discharge_percent = end_discharge.split("%")[0]
                    termination_dict["capacity"] = (float(end_discharge_percent), "%")
                elif end_discharge.endswith("Ah"):
                    end_discharge_Ah = end_discharge.split("Ah")[0]
                    termination_dict["capacity"] = (float(end_discharge_Ah), "Ah")
                else:
                    raise ValueError(
                        "Capacity termination must be given in the form "
                        "'80%', '4Ah', or '4A.h'"
                    )
            elif term.endswith("V"):
                end_discharge_V = term.split("V")[0]
                termination_dict["voltage"] = (float(end_discharge_V), "V")
            elif any(
                [
                    term.endswith(key)
                    for key in [
                        "hour",
                        "hours",
                        "h",
                        "hr",
                        "minute",
                        "minutes",
                        "m",
                        "min",
                        "second",
                        "seconds",
                        "s",
                        "sec",
                    ]
                ]
            ):
                termination_dict["time"] = _convert_time_to_seconds(term)
            else:
                raise ValueError(
                    "Only capacity or voltage can be provided as a termination reason, "
                    "e.g. '80% capacity', '4 Ah capacity', or '2.5 V'"
                )
        return termination_dict

    def search_tag(self, tag):
        """
        Search for a tag in the experiment and return the cycles in which it appears.

        Parameters
        ----------
        tag : str
            The tag to search for

        Returns
        -------
        list
            A list of cycles in which the tag appears
        """
        cycles = []
        for i, cycle in enumerate(self.cycles):
            for step in cycle:
                if tag in step.tags:
                    cycles.append(i)
                    break

        return cycles

    @staticmethod
    def _set_next_start_time(steps):
        end_time = None
        next_start_time = None

        # Loop over the steps in reverse order, setting the end time of each step to the
        # start time of the next step
        for step in reversed(steps):
            step.next_start_time = next_start_time
            step.end_time = end_time

            next_start_time = step.start_time
            if next_start_time:
                end_time = next_start_time

        return steps
