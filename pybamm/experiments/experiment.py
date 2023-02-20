#
# Experiment class
#

import numpy as np
import re


examples = """

    "Discharge at 1C for 0.5 hours at 27oC",
    "Discharge at C/20 for 0.5 hours at 29oC",
    "Charge at 0.5 C for 45 minutes at -5oC",
    "Discharge at 1 A for 0.5 hours at -5.1oC",
    "Charge at 200 mA for 45 minutes at 10.2oC (1 minute period)",
    "Discharge at 1W for 0.5 hours at -10.4oC",
    "Charge at 200mW for 45 minutes",
    "Rest for 10 minutes (5 minute period)",
    "Hold at 1V for 20 seconds",
    "Charge at 1 C until 4.1V",
    "Hold at 4.1 V until 50mA",
    "Hold at 3V until C/50",
    "Discharge at C/3 for 2 hours or until 2.5 V at 26oC",
    "Run US06 (A) at -5oC",
    "Run US06 (V) for 5 minutes",
    "Run US06 (W) for 0.5 hours",

    """


class Experiment:
    """
    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be of the form "Do this for this long" or "Do this until this happens". For example,
    "Charge at 1 C for 1 hour", or "Charge at 1 C until 4.2 V", or "Charge at 1 C for 1
    hour or until 4.2 V at 25oC". The instructions can be of the form
    "(Dis)charge at x A/C/W", "Rest", or "Hold at x V until y A at z oC". The running
    time should be a time in seconds, minutes or
    hours, e.g. "10 seconds", "3 minutes" or "1 hour". The stopping conditions should be
    a circuit state, e.g. "1 A", "C/50" or "3 V". The parameter drive_cycles is
    mandatory to run drive cycle. For example, "Run x", then x must be the key
    of drive_cycles dictionary. The temperature should be provided after the stopping
    condition but before the period, e.g. "1 A at 25 oC (1 second period)". It is
    not essential to provide a temperature and a global temperature can be set either
    from within the paramter values of passing a temperature to this experiment class.
    If the temperature is not specified in a line, then the global temperature is used,
    even if another temperature has been set in an earlier line.

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
    drive_cycles : dict
        Dictionary of drive cycles to use for this experiment.
    cccv_handling : str, optional
        How to handle CCCV. If "two-step" (default), then the experiment is run in
        two steps (CC then CV). If "ode", then the experiment is run in a single step
        using an ODE for current: see
        :class:`pybamm.external_circuit.CCCVFunctionControl` for details.
    """

    def __init__(
        self,
        operating_conditions,
        period="1 minute",
        temperature=None,
        termination=None,
        drive_cycles={},
        cccv_handling="two-step",
    ):
        if cccv_handling not in ["two-step", "ode"]:
            raise ValueError("cccv_handling should be either 'two-step' or 'ode'")
        self.cccv_handling = cccv_handling
        # Save arguments for copying
        self.args = (
            operating_conditions,
            period,
            temperature,
            termination,
            drive_cycles,
            cccv_handling,
        )

        self.period = self.convert_time_to_seconds(period.split())
        self.temperature = temperature

        operating_conditions_cycles = []
        for cycle in operating_conditions:
            # Check types and convert strings to 1-tuples
            if (isinstance(cycle, tuple) or isinstance(cycle, str)) and all(
                [isinstance(cond, str) for cond in cycle]
            ):
                if isinstance(cycle, str):
                    processed_cycle = (cycle,)
                else:
                    processed_cycle = []
                    idx = 0
                    finished = False
                    while not finished:
                        step = cycle[idx]
                        if idx < len(cycle) - 1:
                            next_step = cycle[idx + 1]
                        else:
                            next_step = None
                            finished = True
                        if self.is_cccv(step, next_step):
                            processed_cycle.append(
                                f"{step} then {next_step[0].lower()}{next_step[1:]}"
                            )
                            idx += 2
                        else:
                            processed_cycle.append(step)
                            idx += 1
                        if idx >= len(cycle):
                            finished = True
                operating_conditions_cycles.append(tuple(processed_cycle))
            else:
                try:
                    # Condition is not a string
                    badly_typed_conditions = [
                        cond for cond in cycle if not isinstance(cond, str)
                    ]
                except TypeError:
                    # Cycle is not a tuple or string
                    badly_typed_conditions = []
                badly_typed_conditions = badly_typed_conditions or [cycle]
                raise TypeError(
                    "Operating conditions should be strings or tuples of strings, not "
                    f"{type(badly_typed_conditions[0])}. For example: {examples}"
                )
        self.cycle_lengths = [len(cycle) for cycle in operating_conditions_cycles]
        operating_conditions = [
            cond for cycle in operating_conditions_cycles for cond in cycle
        ]
        self.operating_conditions_cycles = operating_conditions_cycles
        self.operating_conditions_strings = operating_conditions
        self.operating_conditions = [
            self.read_string(cond, drive_cycles) for cond in operating_conditions
        ]

        self.termination_string = termination
        self.termination = self.read_termination(termination)

    def __str__(self):
        return str(self.operating_conditions_cycles)

    def copy(self):
        return Experiment(*self.args)

    def __repr__(self):
        return "pybamm.Experiment({!s})".format(self)

    def read_string(self, cond, drive_cycles):
        """
        Convert a string to a dictionary of the right format

        Parameters
        ----------
        cond : str
            String of appropriate form for example "Charge at x C for y hours". x and y
            must be numbers, 'C' denotes the unit of the external circuit (can be A for
            current, C for C-rate, V for voltage or W for power), and 'hours' denotes
            the unit of time (can be second(s), minute(s) or hour(s))
        drive_cycles: dict
            A map specifying the drive cycles
        """
        if " then " in cond:
            # If the string contains " then ", then this is a two-step CCCV experiment
            # and we need to split it into two strings
            cond_CC, cond_CV = cond.split(" then ")
            op_CC = self.read_string(cond_CC, drive_cycles)
            op_CV = self.read_string(cond_CV, drive_cycles)

            if op_CC["temperature"] != op_CV["temperature"]:
                raise ValueError(
                    "The temperature for the CC and CV steps must be the same."
                    f"Got {op_CC['temperature']} and {op_CV['temperature']}"
                    f"from {op_CC['string']} and {op_CV['string']}"
                )

            tag_CC = op_CC["tags"] or []
            tag_CV = op_CV["tags"] or []
            tags = list(np.unique(tag_CC + tag_CV))
            if len(tags) == 0:
                tags = None

            outputs = {
                "type": "CCCV",
                "Voltage input [V]": op_CV["Voltage input [V]"],
                "time": op_CV["time"],
                "period": op_CV["period"],
                "temperature": op_CC["temperature"],
                "dc_data": None,
                "string": cond,
                "events": op_CV["events"],
                "tags": tags,
            }
            if "Current input [A]" in op_CC:
                outputs["Current input [A]"] = op_CC["Current input [A]"]
            else:
                outputs["C-rate input [-]"] = op_CC["C-rate input [-]"]
            return outputs

        # Read tags
        if " [" in cond:
            cond, tag_str = cond.split(" [")
            tags = tag_str[0:-1].split(",")
        else:
            tags = None

        # Read period
        if " period)" in cond:
            cond, time_period = cond.split(" (")
            time, _ = time_period.split(" period)")
            period = self.convert_time_to_seconds(time.split())
        else:
            period = self.period

        # Temperature part of the condition is removed here
        unprocessed_cond = cond
        temperature, cond = self._read_and_drop_temperature(cond)

        # Read instructions
        if "Run" in cond:
            cond_list = cond.split()
            if "at" in cond:
                raise ValueError(f"Instruction must be of the form: {examples}")
            dc_types = ["(A)", "(V)", "(W)"]
            if all(x not in cond for x in dc_types):
                raise ValueError(
                    "Type of drive cycle must be specified using '(A)', '(V)' or '(W)'."
                    f" For example: {examples}"
                )
            else:
                dc_data = drive_cycles[cond_list[1]]
                # Check for Events
                if "for" in cond:
                    # e.g. for 3 hours
                    idx = cond_list.index("for")
                    end_time = self.convert_time_to_seconds(cond_list[idx + 1 :])
                    ext_drive_cycle = self.extend_drive_cycle(dc_data, end_time)
                    # Drive cycle as numpy array
                    dc_data = ext_drive_cycle
            # Set time and period to 1 second for first step and
            # then calculate the difference in consecutive time steps
            time = dc_data[:, 0][-1]
            period = np.min(np.diff(dc_data[:, 0]))
            # Find the unit of drive cycle ("A", "V", or "W")
            unit = cond_list[2][1]
            electric = {"dc_data": dc_data, "unit": unit}
            events = None
        else:
            dc_data = None
            if "for" in cond and "or until" in cond:
                # e.g. for 3 hours or until 4.2 V
                cond_list = cond.split()
                idx_for = cond_list.index("for")
                idx_until = cond_list.index("or")

                electric = self.convert_electric(cond_list[:idx_for])

                time = self.convert_time_to_seconds(cond_list[idx_for + 1 : idx_until])
                events = self.convert_electric(cond_list[idx_until + 2 :])

            elif "for" in cond:
                # e.g. for 3 hours
                cond_list = cond.split()
                idx = cond_list.index("for")

                electric = self.convert_electric(cond_list[:idx])
                time = self.convert_time_to_seconds(cond_list[idx + 1 :])
                events = None
            elif "until" in cond:
                # e.g. until 4.2 V
                cond_list = cond.split()
                idx = cond_list.index("until")
                electric = self.convert_electric(cond_list[:idx])
                time = None
                events = self.convert_electric(cond_list[idx + 1 :])
            else:
                raise ValueError(
                    "Operating conditions must contain keyword 'for' or 'until' or "
                    f"'Run'. For example: {examples}"
                )

        self.unit_to_type(electric)
        self.unit_to_type(events)

        return {
            **electric,
            "time": time,
            "period": period,
            "temperature": temperature,
            "dc_data": dc_data,
            "string": unprocessed_cond,
            "events": events,
            "tags": tags,
        }

    def unit_to_type(self, electric):
        if electric is not None:
            unit = electric.pop("unit")
            if unit == "C":
                electric["type"] = "C-rate"
            elif unit == "A":
                electric["type"] = "current"
            elif unit == "V":
                electric["type"] = "voltage"
            elif unit == "W":
                electric["type"] = "power"
            return electric

    def extend_drive_cycle(self, drive_cycle, end_time):
        "Extends the drive cycle to enable for event"
        temp_time = []
        temp_time.append(drive_cycle[:, 0])
        loop_end_time = temp_time[0][-1]
        i = 1
        while loop_end_time <= end_time:
            # Extend the drive cycle until the drive cycle time
            # becomes greater than specified end time
            temp_time.append(
                np.append(temp_time[i - 1], temp_time[0] + temp_time[i - 1][-1] + 1)
            )
            loop_end_time = temp_time[i][-1]
            i += 1
        time = temp_time[-1]
        drive_data = np.tile(drive_cycle[:, 1], i)
        # Combine the drive cycle time and data
        ext_drive_cycle = np.column_stack((time, drive_data))
        # Limit the drive cycle to the specified end_time
        ext_drive_cycle = ext_drive_cycle[ext_drive_cycle[:, 0] <= end_time]
        del temp_time
        return ext_drive_cycle

    def convert_electric(self, electric):
        """Convert electrical instructions to consistent output"""
        # Rest == zero current
        if electric[0].lower() == "rest":
            return {"Current input [A]": 0, "unit": "A"}
        else:
            if len(electric) in [3, 4]:
                if len(electric) == 4:
                    # e.g. Charge at 4 A, Hold at 3 V
                    instruction, _, value, unit = electric
                    value_unit = value + unit
                elif len(electric) == 3:
                    # e.g. Discharge at C/2, Charge at 1A
                    instruction, _, value_unit = electric
                # Read instruction
                if instruction.lower() in ["discharge", "hold"]:
                    sign = 1
                elif instruction.lower() == "charge":
                    sign = -1
                else:
                    raise ValueError(
                        "Instruction must be 'discharge', 'charge', 'rest', 'hold' or "
                        f"'Run'. For example: {examples}"
                        ""
                        "The following instruction does not comply: "
                        f"{instruction}"
                    )
            elif len(electric) == 2:
                # e.g. 3 A, 4.1 V
                value, unit = electric
                value_unit = value + unit
                sign = 1
            elif len(electric) == 1:
                # e.g. C/2, 1A
                value_unit = electric[0]
                sign = 1
            else:
                raise ValueError(
                    """Instruction '{}' not recognized. Some acceptable examples are: {}
                    """.format(
                        " ".join(electric), examples
                    )
                )
            # e.g. C/2, 1A
            if value_unit[0] == "C":
                # e.g. C/2
                unit = value_unit[0]
                value = 1 / float(value_unit[2:])
            else:
                unit = value_unit[-1:]
                if "m" in value_unit:
                    # e.g. 1mA
                    value = float(value_unit[:-2]) / 1000
                else:
                    # e.g. 1A
                    value = float(value_unit[:-1])
            # Read value and units
            if unit == "C":
                return {"C-rate input [-]": sign * float(value), "unit": unit}
            elif unit == "A":
                return {"Current input [A]": sign * float(value), "unit": unit}
            elif unit == "V":
                return {"Voltage input [V]": float(value), "unit": unit}
            elif unit == "W":
                return {"Power input [W]": sign * float(value), "unit": unit}
            else:
                raise ValueError(
                    """units must be 'C', 'A', 'mA', 'V', 'W' or 'mW', not '{}'.
                    For example: {}
                    """.format(
                        unit, examples
                    )
                )

    def _read_and_drop_temperature(self, cond):
        matches = re.findall(r"at\s-*\d+\.*\d*\s*oC", cond)

        if len(matches) == 0:
            if "oC" in cond:
                raise ValueError(f"Temperature not written correctly on step: '{cond}'")
            temperature = self.temperature
            reduced_cond = cond

        elif len(matches) == 1:
            match = matches[0]
            numerical_part = re.findall(r"-*\d+\.*\d*", match)[0]
            temperature = float(numerical_part)
            reduced_cond = cond.replace(match, "")

        else:
            raise ValueError(f"More than one temperature found on step: '{cond}'")

        return temperature, reduced_cond

    def convert_time_to_seconds(self, time_and_units):
        """Convert a time in seconds, minutes or hours to a time in seconds"""
        time, units = time_and_units
        if units in ["second", "seconds", "s", "sec"]:
            time_in_seconds = float(time)
        elif units in ["minute", "minutes", "m", "min"]:
            time_in_seconds = float(time) * 60
        elif units in ["hour", "hours", "h", "hr"]:
            time_in_seconds = float(time) * 3600
        else:
            raise ValueError(
                """time units must be 'seconds', 'minutes' or 'hours'. For example: {}
                """.format(
                    examples
                )
            )
        return time_in_seconds

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
                if end_discharge.endswith("%"):
                    end_discharge_percent = end_discharge.split("%")[0]
                    termination_dict["capacity"] = (float(end_discharge_percent), "%")
                elif end_discharge.endswith("Ah"):
                    end_discharge_Ah = end_discharge.split("Ah")[0]
                    termination_dict["capacity"] = (float(end_discharge_Ah), "Ah")
                elif end_discharge.endswith("A.h"):
                    end_discharge_Ah = end_discharge.split("A.h")[0]
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

    def is_cccv(self, step, next_step):
        """
        Check whether a step and the next step indicate a CCCV charge
        """
        if self.cccv_handling == "two-step" or next_step is None:
            return False
        # e.g. step="Charge at 2.0 A until 4.2V"
        # next_step="Hold at 4.2V until C/50"
        if (
            (step.startswith("Charge") or step.startswith("Discharge"))
            and "until" in step
            and "V" in step
            and "Hold at " in next_step
            and "V until" in next_step
        ):
            op = self.read_string(step, None)
            next_op = self.read_string(next_step, None)
            # Check that the event conditions are the same as the hold conditions
            if op["events"] == {k: v for k, v in next_op.items() if k in op["events"]}:
                return True
        return False

    def search_tag(self, tag):
        cycles = []
        for i, cycle in enumerate(self.operating_conditions_cycles):
            for cond in cycle:
                if " [" in cond:
                    cond, tag_str = cond.split(" [")
                    tags = tag_str[0:-1].split(",")
                    if tag in tags:
                        cycles.append(i)
                        break

        return cycles
