from warnings import warn

import pybamm


class BaseTermination:
    """
    Base class for a termination event for an experiment step. To create a custom
    termination, a class must implement `get_event_expression` to return the symbolic
    expression for the event. In most cases the class
    :class:`pybamm.step.CustomTermination` can be used to assist with this.

    Parameters
    ----------
    value : float
        The value at which the event is triggered
    """

    def __init__(self, value, operator=None):
        self.value = value
        if operator not in ["<", ">", None]:
            raise ValueError(f"Invalid operator: {operator}")
        self.operator = operator

    def get_event_name(self, step):
        raise NotImplementedError  # pragma: no cover

    def get_event_expression(self, variables, step):
        """
        Return the symbolic expression corresponding to the termination event.

        Parameters
        ----------
        variables : dict
            Dictionary of model variables, to be used for selecting the variable(s) that
            determine the event
        step : :class:`pybamm.step.BaseStep`
            Step for which this is a termination event, to be used in some
            cases to determine the sign of the event.
        """
        raise NotImplementedError  # pragma: no cover

    def get_event(self, variables, step):
        expression = self.get_event_expression(variables, step)
        if expression is None:
            return None
        return pybamm.Event(self.get_event_name(step), expression)

    def __eq__(self, other):
        # objects are equal if they have the same type and value
        if isinstance(other, self.__class__):
            return self.value == other.value
        else:
            return False


class CRateTermination(BaseTermination):
    """
    Termination based on C-rate, created when a string termination of the C-rate type
    (e.g. "C/10") is provided
    """

    def get_event_name(self, step):
        return "C-rate cut-off [experiment]"

    def get_event_expression(self, variables, step):
        return abs(variables["C-rate"]) - self.value


class CrateTermination(CRateTermination):
    """
    Termination based on C-rate, created when a string termination of the C-rate type
    (e.g. "C/10") is provided
    """

    def __init__(self, value, operator=None):
        super().__init__(value, operator)
        warning = DeprecationWarning(
            "CrateTermination is deprecated and will be removed in a future release. Use CRateTermination instead."
        )
        warn(warning, stacklevel=2)


class CurrentTermination(BaseTermination):
    """
    Termination based on current, created when a string termination of the current type
    (e.g. "1A") is provided
    """

    def get_event_name(self, step):
        operator = self.operator
        if operator == ">":
            return f"Current [A] > {self.value} [A] [experiment]"
        elif operator == "<":
            return f"Current [A] < {self.value} [A] [experiment]"
        else:
            return f"abs(Current [A]) < {self.value} [A] [experiment]"

    def get_event_expression(self, variables, step):
        operator = self.operator
        if operator == ">":
            return self.value - variables["Current [A]"]
        elif operator == "<":
            return variables["Current [A]"] - self.value
        else:
            return abs(variables["Current [A]"]) - self.value


class VoltageTermination(BaseTermination):
    """
    Termination based on voltage, created when a string termination of the voltage type
    (e.g. "4.2V") is provided
    """

    def _get_operator(self, step):
        operator = self.operator
        if operator is None:
            direction = step.direction
            if direction == "charge":
                operator = ">"
            elif direction == "discharge":
                operator = "<"
            else:
                return None
        return operator

    def get_event_name(self, step):
        operator = self._get_operator(step)
        if operator is None:
            return None
        return f"Voltage {operator} {self.value} [V] [experiment]"

    def get_event_expression(self, variables, step):
        # The voltage event should be positive at the start of charge/
        # discharge. We use the sign of the current or power input to
        # figure out whether the voltage event is greater than the starting
        # voltage (charge) or less (discharge) and set the sign of the
        # event accordingly
        operator = self._get_operator(step)
        if operator is None:
            return None

        if operator == ">":
            sign = -1
        else:
            sign = 1

        return sign * (variables["Battery voltage [V]"] - self.value)


class Voltage:
    def __gt__(self, value):
        return VoltageTermination(value, operator=">")

    def __lt__(self, value):
        return VoltageTermination(value, operator="<")


class Current:
    def __gt__(self, value):
        return CurrentTermination(value, operator=">")

    def __lt__(self, value):
        return CurrentTermination(value, operator="<")


class CustomTermination(BaseTermination):
    """
    Define a custom termination event using a function. This can be used to create an
    event based on any variable in the model.

    Parameters
    ----------
    name : str
        Name of the event
    event_function : callable
        A function that takes in a dictionary of variables and evaluates the event
        value. Must be positive before the event is triggered and zero when the
        event is triggered.

    Example
    -------
    Add a cut-off based on negative electrode stoichiometry. The event will trigger
    when the negative electrode stoichiometry reaches 10%.

    >>> def neg_stoich_cutoff(variables):
    ...    return variables["Negative electrode stoichiometry"] - 0.1

    >>> neg_stoich_termination = pybamm.step.CustomTermination(
    ...    name="Negative stoichiometry cut-off", event_function=neg_stoich_cutoff
    ... )
    """

    def __init__(self, name, event_function):
        if not name.endswith(" [experiment]"):
            name += " [experiment]"
        self.name = name
        self.event_function = event_function

    def get_event_name(self, step):
        return self.name

    def get_event_expression(self, variables, step):
        return self.event_function(variables)


def _read_termination(termination, operator=None):
    if isinstance(termination, tuple):
        op, typ, value = termination
    else:
        return termination

    termination_class = {
        "current": CurrentTermination,
        "voltage": VoltageTermination,
        "C-rate": CRateTermination,
    }[typ]
    return termination_class(value, operator=op)
