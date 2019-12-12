#
# Experiment class
#


class Experiment:
    """
    Base class for experimental conditions under which to run the model

    Parameters
    ----------
    operating_conditions : list
        List of operating conditions

    Examples
    --------
    >>> experiment = pybamm.Experiment(["1C for 0.5 hours", "0.5C for 45 minutes"])
    """

    def __init__(self, operating_conditions):
        self.operating_conditions_string = str(operating_conditions)
        self.operating_conditions = self.read_operating_conditions(operating_conditions)

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
        for cond in operating_conditions:
            if isinstance(cond, str):
                converted_operating_conditions.append(self.str_to_tuple(cond))
            else:
                raise TypeError(
                    "Conditions should be tuples or strings, not {}".format(type(cond))
                )

        return converted_operating_conditions

    def str_to_tuple(self, cond):
        """
        Convert a string to a tuple of the right format

        Parameters
        ----------
        cond : str
            String of appropriate form for example "x C for y hours". x and y must be
            numbers, 'C' denotes the unit of the external circuit (can be A for current, 
            C for C-rate, V for voltage or W for power), and 'hours' denotes the unit of
            time (can be second(s), minute(s) or hour(s))
        """
        cond_tuple = cond.split()
        self.check_tuple_condition(cond_tuple)
        cond_tuple = self.convert_time_to_seconds(cond_tuple)
        return cond_tuple

    def check_tuple_condition(self, cond_tuple):
        "Check tuple of conditions has the right form"
        # Check length
        if len(cond_tuple) != 5:
            raise ValueError(
                "Tuple operating conditions should have length 5, but is {}".format(
                    cond_tuple
                )
            )
        # Check inputs
        try:
            float(cond_tuple[0])
        except ValueError:
            raise TypeError(
                """ First entry in a tuple of conditions should be a number, not {}
                """.format(
                    cond_tuple[0]
                )
            )
        acceptable_strings = ["A", "C", "V", "W"]
        if cond_tuple[1] not in acceptable_strings:
            raise ValueError(
                """Second entry in a tuple of conditions should be one of {} but is {}
                """.format(
                    acceptable_strings, cond_tuple[1]
                )
            )
        if cond_tuple[2] != "for":
            raise ValueError(
                "Third entry in a tuple of conditions should be 'for', not {}".format(
                    cond_tuple[2]
                )
            )
        try:
            float(cond_tuple[3])
        except ValueError:
            raise TypeError("Fourth entry in a tuple of conditions should be a number")

    def convert_time_to_seconds(self, cond_tuple):
        "Convert a time in seconds, minutes or hours to a time in seconds"
        time, units = cond_tuple[3:]
        if units in ["second", "seconds"]:
            time_in_seconds = float(time)
        elif units in ["minute", "minutes"]:
            time_in_seconds = float(time) * 60
        elif units in ["hour", "hours"]:
            time_in_seconds = float(time) * 3600
        return tuple([float(cond_tuple[0]), cond_tuple[1], time_in_seconds])

