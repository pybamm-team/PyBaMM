#
# Experiment class
#

examples = """

    Discharge at 1 C for 0.5 hours,
    Discharge at C/20 for 0.5 hours,
    Charge at 0.5 C for 45 minutes,
    Discharge at 1 A for 90 seconds,
    Charge at 200 mA for 45 minutes,
    Discharge at 1 W for 0.5 hours,
    Charge at 200 mW for 45 minutes,
    Rest for 10 minutes,
    Hold at 1 V for 20 seconds,
    Charge at 1 C until 4.1 V,
    Hold at 4.1 V until 50 mA,
    """


class Experiment:
    """
    Base class for experimental conditions under which to run the model

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions
    parameters : dict
        Dictionary of parameters to use for this experiment, replacing default
        parameters as appropriate
    frequency : string, optional
        Frequency at which to record outputs. Default is 1 minute.

    Examples
    --------
    >>> experiment = pybamm.Experiment(["1C for 0.5 hours", "0.5C for 45 minutes"])
    """

    def __init__(self, operating_conditions, parameters=None, frequency="1 minute"):
        self.operating_conditions_string = operating_conditions
        self.operating_conditions, self.events = self.read_operating_conditions(
            operating_conditions
        )
        parameters = parameters or {}
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError("experimental parameters should be a dictionary")
        self.frequency = self.convert_time_to_seconds(frequency.split())

    def __str__(self):
        return self.operating_conditions_string

    def __repr__(self):
        return "pybamm.Experiment({})".format(self.operating_conditions_string)

    def read_operating_conditions(self, operating_conditions):
        """
        Convert operating conditions to the appropriate format

        Parameters
        ----------
        operating_conditions : list
            List of operating conditions

        Returns
        -------
        operating_conditions : list
            Operating conditions in the tuple format
        """
        converted_operating_conditions = []
        events = []
        for cond in operating_conditions:
            if isinstance(cond, str):
                next_op, next_event = self.read_string(cond)
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

    def read_string(self, cond):
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
        cond_list = cond.split()
        if "for" in cond_list:
            idx = cond_list.index("for")
            electric = self.convert_electric(cond_list[:idx])
            time = self.convert_time_to_seconds(cond_list[idx + 1 :])
            events = None
        elif "until" in cond_list:
            idx = cond_list.index("until")
            electric = self.convert_electric(cond_list[:idx])
            time = None
            events = self.convert_electric(cond_list[idx + 1 :])
        else:
            raise ValueError(
                """Operating conditions must contain keyword 'for' or 'until'.
                For example: {}""".format(
                    examples
                )
            )
        return electric + (time,), events

    def convert_electric(self, electric):
        "Convert electrical instructions to consistent output"
        # Rest == zero current
        if electric[0].lower() == "rest":
            return (0, "A")
        else:
            if len(electric) in [3, 4]:
                if len(electric) == 4:
                    instruction, _, value, unit = electric
                elif len(electric) == 3:
                    instruction, _, value_unit = electric
                    unit = value_unit[0]
                    value = 1 / float(value_unit[2:])
                # Read instruction
                if instruction.lower() in ["discharge", "hold"]:
                    sign = 1
                elif instruction.lower() == "charge":
                    sign = -1
                else:
                    raise ValueError(
                        """instruction must be 'discharge', 'charge', 'rest' or 'hold'.
                        For example: {}""".format(
                            examples
                        )
                    )
            elif len(electric) == 2:
                value, unit = electric
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
                    """units must be 'C', 'A', 'mA', 'V', 'W' or 'mW'. For example: {}
                    """.format(
                        examples
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


if __name__ == "__main__":
    Experiment(["Rest for 10 bla"])
