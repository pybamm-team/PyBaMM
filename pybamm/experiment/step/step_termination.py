import pybamm
import numpy as np


class BaseTermination:
    """
    Base class for a termination event for an experiment step. To create a custom
    termination, a class must implement `get_event` to return a :class:`pybamm.Event`
    corresponding to the desired termination. In most cases the class
    :class:`pybamm.step.CustomTermination` can be used to assist with this.

    Parameters
    ----------
    value : float
        The value at which the event is triggered
    """

    def __init__(self, value):
        self.value = value

    def get_event(self, variables, step_value):
        """
        Return a :class:`pybamm.Event` object corresponding to the termination event

        Parameters
        ----------
        variables : dict
            Dictionary of model variables, to be used for selecting the variable(s) that
            determine the event
        step_value : float or :class:`pybamm.Symbol`
            Value of the step for which this is a termination event, to be used in some
            cases to determine the sign of the event.
        """
        raise NotImplementedError

    def __eq__(self, other):
        # objects are equal if they have the same type and value
        if isinstance(other, self.__class__):
            return self.value == other.value
        else:
            return False


class CrateTermination(BaseTermination):
    """
    Termination based on C-rate, created when a string termination of the C-rate type
    (e.g. "C/10") is provided
    """

    def get_event(self, variables, step_value):
        """
        See :meth:`BaseTermination.get_event`
        """
        event = pybamm.Event(
            "C-rate cut-off [experiment]",
            abs(variables["C-rate"]) - self.value,
        )
        return event


class CurrentTermination(BaseTermination):
    """
    Termination based on current, created when a string termination of the current type
    (e.g. "1A") is provided
    """

    def get_event(self, variables, step_value):
        """
        See :meth:`BaseTermination.get_event`
        """
        event = pybamm.Event(
            "Current cut-off [A] [experiment]",
            abs(variables["Current [A]"]) - self.value,
        )
        return event


class VoltageTermination(BaseTermination):
    """
    Termination based on voltage, created when a string termination of the voltage type
    (e.g. "4.2V") is provided
    """

    def get_event(self, variables, step_value):
        """
        See :meth:`BaseTermination.get_event`
        """
        # The voltage event should be positive at the start of charge/
        # discharge. We use the sign of the current or power input to
        # figure out whether the voltage event is greater than the starting
        # voltage (charge) or less (discharge) and set the sign of the
        # event accordingly
        if isinstance(step_value, pybamm.Symbol):
            inpt = {"start time": 0}
            init_curr = step_value.evaluate(t=0, inputs=inpt).flatten()[0]
        else:
            init_curr = step_value
        sign = np.sign(init_curr)
        if sign > 0:
            name = "Discharge"
        else:
            name = "Charge"
        if sign != 0:
            # Event should be positive at initial conditions for both
            # charge and discharge
            event = pybamm.Event(
                f"{name} voltage cut-off [V] [experiment]",
                sign * (variables["Battery voltage [V]"] - self.value),
            )
            return event


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

    def get_event(self, variables, step_value):
        """
        See :meth:`BaseTermination.get_event`
        """
        return pybamm.Event(self.name, self.event_function(variables))


def _read_termination(termination):
    if isinstance(termination, tuple):
        typ, value = termination
    else:
        return termination

    termination_class = {
        "current": CurrentTermination,
        "voltage": VoltageTermination,
        "C-rate": CrateTermination,
    }[typ]
    return termination_class(value)
