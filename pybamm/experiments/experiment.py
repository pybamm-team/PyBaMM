#
# Experiment class
#

examples = """

    Discharge at 1C for 0.5 hours,
    Discharge at C/20 for 0.5 hours,
    Charge at 0.5 C for 45 minutes,
    Discharge at 1 A for 90 seconds,
    Charge at 200mA for 45 minutes (1 minute period),
    Discharge at 1 W for 0.5 hours,
    Charge at 200 mW for 45 minutes,
    Rest for 10 minutes (5 minute period),
    Hold at 1 V for 20 seconds,
    Charge at 1 C until 4.1V,
    Hold at 4.1 V until 50 mA,
    Hold at 3V until C/50,
    Run US06,
    Run US06 for 20 seconds,
    Run US06 for 45 minutes,
    Run US06 for 2 hours,
    """


class Experiment:
    """
    Base class for experimental conditions under which to run the model. In general, a
    list of operating conditions should be passed in. Each operating condition should
    be of the form "Do this for this long" or "Do this until this happens". For example,
    "Charge at 1 C for 1 hour", or "Charge at 1 C until 4.2 V", or "Charge at 1 C for 1
    hour or until 4.2 V". The instructions can be of the form "(Dis)charge at x A/C/W",
    "Rest", or "Hold at x V". The running time should be a time in seconds, minutes or
    hours, e.g. "10 seconds", "3 minutes" or "1 hour". The stopping conditions should be
    a circuit state, e.g. "1 A", "C/50" or "3 V". The parameter drive_cycles is
    mandatory to run drive cycle. For example, "Run x", then x must be the key
    of drive_cycles dictionary.

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions
    parameters : dict
        Dictionary of parameters to use for this experiment, replacing default
        parameters as appropriate
    period : string, optional
        Period (1/frequency) at which to record outputs. Default is 1 minute. Can be
        overwritten by individual operating conditions.
    drive_cycles : dict
        Dictionary of drive cycles to use for this experiment.
    """

    def __init__(self,
                 operating_conditions,
                 parameters=None,
                 period="1 minute",
                 drive_cycles={}):
        import numpy as np
        self._np = np
        self.period = self.convert_time_to_seconds(period.split())
        self.operating_conditions_strings = operating_conditions
        self.operating_conditions, self.events = self.read_operating_conditions(
            operating_conditions, drive_cycles
        )
        parameters = parameters or {}
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError("experimental parameters should be a dictionary")

    def __str__(self):
        return str(self.operating_conditions_strings)

    def __repr__(self):
        return "pybamm.Experiment({!s})".format(self)

    def read_operating_conditions(self, operating_conditions, drive_cycles):
        """
        Convert operating conditions to the appropriate format

        Parameters
        ----------
        operating_conditions : list
            List of operating conditions
        drive_cycles : dictionary
            Dictionary of Drive Cycles
        Returns
        -------
        operating_conditions : list
            Operating conditions in the tuple format
        """
        converted_operating_conditions = []
        events = []
        for cond in operating_conditions:
            if isinstance(cond, str):
                next_op, next_event = self.read_string(cond, drive_cycles)
                converted_operating_conditions.append(next_op)
                events.append(next_event)
            else:
                raise TypeError(
                    """Operating conditions should be strings, not {}. For example: {}
                    """.format(
                        type(cond), examples
                    )
                )

        return converted_operating_conditions, events

    def read_string(self, cond, drive_cycles):
        """
        Convert a string to a tuple of the right format

        Parameters
        ----------
        cond : str
            String of appropriate form for example "Charge at x C for y hours". x and y
            must be numbers, 'C' denotes the unit of the external circuit (can be A for
            current, C for C-rate, V for voltage or W for power), and 'hours' denotes
            the unit of time (can be second(s), minute(s) or hour(s))
        """
        # Read period
        if " period)" in cond:
            cond, time_period = cond.split("(")
            time, _ = time_period.split(" period)")
            period = self.convert_time_to_seconds(time.split())
        else:
            period = self.period
        # Read instructions
        if "Run" in cond:
            cond_list = cond.split()
            if "at" in cond:
                raise ValueError(
                    """Instruction must be
                    For example: {}""".format(
                        examples
                    )
                )
                # Check for Events
            elif "for" in cond and "or until" in cond:
                # e.g. for 3 hours or until 4.2 V
                idx_for = cond_list.index("for")
                idx_until = cond_list.index("or")
                end_time = self.convert_time_to_seconds(
                    cond_list[idx_for + 1:idx_until])
                ext_drive_cycle = self.extend_drive_cycle(drive_cycles[cond_list[1]],
                                                          end_time)
                electric = (ext_drive_cycle[:, 1], "Drive")
                time = self._np.append(1, self._np.diff(ext_drive_cycle[:, 0]))
                period = time
                events = self.convert_electric(cond_list[idx_until + 2 :])
            elif "for" in cond:
                # e.g. for 3 hours
                idx = cond_list.index("for")
                end_time = self.convert_time_to_seconds(cond_list[idx + 1 :])
                ext_drive_cycle = self.extend_drive_cycle(drive_cycles[cond_list[1]],
                                                          end_time)
                electric = (ext_drive_cycle[:, 1], "Drive")
                time = self._np.append(1, self._np.diff(ext_drive_cycle[:, 0]))
                period = time
                events = None
            elif "until" in cond:
                # e.g. until 4.2 V
                idx = cond_list.index("until")
                ext_drive_cycle = self.extend_drive_cycle(drive_cycles[cond_list[1]])
                electric = (ext_drive_cycle[:, 1], "Drive")
                time = self._np.append(1, self._np.diff(ext_drive_cycle[:, 0]))
                period = time
                events = self.convert_electric(cond_list[idx + 1 :])
            else:
                # e.g. Run US06
                electric = (drive_cycles[cond_list[1]][:, 1], "Drive")
                # Set time and period to 1 second for first step and
                # then calculate the difference in consecutive time steps
                time = self._np.append(1,
                                       self._np.diff(drive_cycles[cond_list[1]][:, 0]))
                period = time
                events = None
        elif "Run" not in cond:
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
                    """Operating conditions must contain keyword 'for' or 'until' or 'Run'.
                    For example: {}""".format(
                        examples
                    )
                )
        return electric + (time,) + (period,), events

    def extend_drive_cycle(self, drive_cycle, end_time=None):
        "Extends the drive cycle to enable until and for events"
        temp_time = []
        temp_time.append(drive_cycle[:, 0])
        loop_end_time = temp_time[0][-1]
        i = 1
        if end_time is None:
            # Extend for 1 week.
            end_time = 7 * 24 * 60 * 60
        while loop_end_time <= end_time:
            # Extend the drive cycle until the drive cycle time
            # becomes greater than specified end time
            temp_time.append(self._np.append(temp_time[i - 1],
                                             temp_time[0] + temp_time[i - 1][-1] + 1))
            loop_end_time = temp_time[i][-1]
            i += 1
        time = temp_time[-1]
        drive_data = self._np.tile(drive_cycle[:, 1], i)
        # Combine the drive cycle time and data
        ext_drive_cycle = self._np.column_stack((time, drive_data))
        # Limit the drive cycle to the specified end_time
        ext_drive_cycle = ext_drive_cycle[ext_drive_cycle[:, 0] <= end_time]
        del temp_time
        return ext_drive_cycle

    def convert_electric(self, electric):
        "Convert electrical instructions to consistent output"
        # Rest == zero current
        if electric[0].lower() == "rest":
            return (0, "A")
        else:
            if len(electric) in [3, 4]:
                if len(electric) == 4:
                    # e.g. Charge at 4 A, Hold at 3 V
                    instruction, _, value, unit = electric
                elif len(electric) == 3:
                    # e.g. Discharge at C/2, Charge at 1A
                    instruction, _, value_unit = electric
                    if value_unit[0] == "C":
                        # e.g. C/2
                        unit = value_unit[0]
                        value = 1 / float(value_unit[2:])
                    else:
                        # e.g. 1A
                        if "m" in value_unit:
                            # e.g. 1mA
                            unit = value_unit[-2:]
                            value = float(value_unit[:-2])
                        else:
                            # e.g. 1A
                            unit = value_unit[-1]
                            value = float(value_unit[:-1])
                # Read instruction
                if instruction.lower() in ["discharge", "hold"]:
                    sign = 1
                elif instruction.lower() == "charge":
                    sign = -1
                else:
                    raise ValueError(
                        """instruction must be 'discharge', 'charge', 'rest', 'hold' or 'Run'.
                        For example: {}""".format(
                            examples
                        )
                    )
            elif len(electric) == 2:
                # e.g. 3 A, 4.1 V
                value, unit = electric
                sign = 1
            elif len(electric) == 1:
                # e.g. C/2, 1A
                value_unit = electric[0]
                if value_unit[0] == "C":
                    # e.g. C/2
                    unit = value_unit[0]
                    value = 1 / float(value_unit[2:])
                else:
                    if "m" in value_unit:
                        # e.g. 1mA
                        unit = value_unit[-2:]
                        value = float(value_unit[:-2])
                    else:
                        # e.g. 1A
                        unit = value_unit[-1]
                        value = float(value_unit[:-1])
                sign = 1
            else:
                raise ValueError(
                    """Instruction '{}' not recognized. Some acceptable examples are: {}
                    """.format(
                        " ".join(electric), examples
                    )
                )
            # Read value and units
            if unit == "C":
                return (sign * float(value), "C")
            elif unit == "A":
                return (sign * float(value), "A")
            elif unit == "mA":
                return (sign * float(value) / 1000, "A")
            elif unit == "V":
                return (float(value), "V")
            elif unit == "W":
                return (sign * float(value), "W")
            elif unit == "mW":
                return (sign * float(value) / 1000, "W")
            else:
                raise ValueError(
                    """units must be 'C', 'A', 'mA', 'V', 'W' or 'mW', not '{}'.
                    For example: {}
                    """.format(
                        unit, examples
                    )
                )

    def convert_time_to_seconds(self, time_and_units):
        "Convert a time in seconds, minutes or hours to a time in seconds"
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
