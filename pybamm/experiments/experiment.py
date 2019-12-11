#
# Experiment class
#
import numbers


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
        self.operating_conditions_string = operating_conditions
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
        for i, cond in enumerate(operating_conditions):
            if isinstance(cond, str):
                operating_conditions[i] = self.str_to_tuple(cond)
            else:
                raise TypeError(
                    "Conditions should be tuples or strings, not {}".format(type(cond))
                )

        return operating_conditions

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
        cond_tuple = self.to_seconds(cond_tuple)
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
        if not isinstance(cond_tuple[0], numbers.Number):
            raise TypeError("First entry in a tuple of conditions should be a number")
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
        if not isinstance(cond_tuple[3], numbers.Number):
            raise TypeError("Fourth entry in a tuple of conditions should be a number")

