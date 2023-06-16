#
# Experiment class
#

import pybamm
from pybamm.step._steps_util import (
    _convert_time_to_seconds,
    _convert_temperature_to_kelvin,
)


class Experiment:
    """
    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be either a `pybamm.step._Step` class, created using one of the methods
    `pybamm.step.current`, `pybamm.step.c_rate`, `pybamm.step.voltage`
    , `pybamm.step.power`, `pybamm.step.resistance`, or
    `pybamm.step.string`, or a string, in which case the string is passed to
    `pybamm.step.string`.

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions
    period : string, optional
        Period (1/frequency) at which to record outputs. Default is 1 minute. Can be
        overwritten by individual operating conditions.
    temperature: float, optional
        The ambient air temperature in degrees Celsius at which to run the experiment.
        Default is None whereby the ambient temperature is taken from the parameter set.
        This value is overwritten if the temperature is specified in a step.
    termination : list, optional
        List of conditions under which to terminate the experiment. Default is None.
        This is different from the termination for individual steps. Termination for
        individual steps is specified in the step itself, and the simulation moves to
        the next step when the termination condition is met
        (e.g. 2.5V discharge cut-off). Termination for the
        experiment as a whole is specified here, and the simulation stops when the
        termination condition is met (e.g. 80% capacity).
    """

    def __init__(
        self,
        operating_conditions,
        period="1 minute",
        temperature=None,
        termination=None,
        drive_cycles=None,
        cccv_handling=None,
    ):
        if cccv_handling is not None:
            raise ValueError(
                "cccv_handling has been deprecated, use "
                "`pybamm.step.cccv_ode(current, voltage)` instead to produce the "
                "same behavior as the old `cccv_handling='ode'`"
            )
        if drive_cycles is not None:
            raise ValueError(
                "drive_cycles should now be passed as an experiment step object, e.g. "
                "`pybamm.step.current(drive_cycle)`"
            )
        # Save arguments for copying
        self.args = (
            operating_conditions,
            period,
            temperature,
            termination,
        )

        self.datetime_formats = [
            "Day %j %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

        operating_conditions_cycles = []
        for cycle in operating_conditions:
            # Check types and convert to list
            if not isinstance(cycle, tuple):
                cycle = (cycle,)
            operating_conditions_cycles.append(cycle)

        self.operating_conditions_cycles = operating_conditions_cycles
        self.cycle_lengths = [len(cycle) for cycle in operating_conditions_cycles]

        operating_conditions_steps_unprocessed = self._set_next_start_time(
            [cond for cycle in operating_conditions_cycles for cond in cycle]
        )

        # Convert strings to pybamm.step._Step objects
        # We only do this once per unique step, do avoid unnecessary conversions
        unique_steps_unprocessed = set(operating_conditions_steps_unprocessed)
        processed_steps = {}
        for step in unique_steps_unprocessed:
            if isinstance(step, str):
                processed_steps[step] = pybamm.step.string(step)
            elif isinstance(step, pybamm.step._Step):
                processed_steps[step] = step

        # Save the processed unique steps and the processed operating conditions
        # for every step
        self.unique_steps = set(processed_steps.values())
        self.operating_conditions_steps = [
            processed_steps[step] for step in operating_conditions_steps_unprocessed
        ]

        self.initial_start_time = self.operating_conditions_steps[0].start_time

        if (
            self.operating_conditions_steps[0].end_time is not None
            and self.initial_start_time is None
        ):
            raise ValueError(
                "When using experiments with `start_time`, the first step must have a "
                "`start_time`."
            )

        self.termination_string = termination
        self.termination = self.read_termination(termination)

        # Modify steps with period and temperature in place
        self.period = _convert_time_to_seconds(period)
        self.temperature = _convert_temperature_to_kelvin(temperature)
        for step in self.unique_steps:
            if step.period is None:
                step.period = self.period
            if step.temperature is None:
                step.temperature = self.temperature

    def __str__(self):
        return str(self.operating_conditions_cycles)

    def copy(self):
        return Experiment(*self.args)

    def __repr__(self):
        return "pybamm.Experiment({!s})".format(self)

    def read_termination(self, termination):
        """
        Read the termination reason. If this condition is hit, the experiment will stop.
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
        for i, cycle in enumerate(self.operating_conditions_cycles):
            for step in cycle:
                if tag in step.tags:
                    cycles.append(i)
                    break

        return cycles

    def _set_next_start_time(self, operating_conditions):
        if all(isinstance(i, str) for i in operating_conditions):
            return operating_conditions

        end_time = None
        next_start_time = None

        for op in reversed(operating_conditions):
            if isinstance(op, str):
                op = pybamm.step.string(op)
            elif not isinstance(op, pybamm.step._Step):
                raise TypeError(
                    "Operating conditions should be strings or _Step objects"
                )

            op.next_start_time = next_start_time
            op.end_time = end_time

            next_start_time = op.start_time
            if next_start_time:
                end_time = next_start_time

        return operating_conditions
