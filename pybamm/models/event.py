from enum import Enum


class EventType(Enum):
    """
    Defines the type of event, see :class:`pybamm.Event`

    TERMINATION indicates an event that will terminate the solver, the expression should
    return 0 when the event is triggered

    DISCONTINUITY indicates an expected discontinuity in the solution, the expression
    should return the time that the discontinuity occurs. The solver will integrate up
    to the discontinuity and then restart just after the discontinuity.

    """

    TERMINATION = 0
    DISCONTINUITY = 1


class Event:
    """

    Defines an event for use within a pybamm model

    Attributes
    ----------

    name: str
        A string giving the name of the event
    event_type: :class:`pybamm.EventType`
        An enum defining the type of event
    expression: :class:`pybamm.Symbol`
        An expression that defines when the event occurs


    """

    def __init__(self, name, expression, event_type=EventType.TERMINATION):
        self._name = name
        self._expression = expression
        self._event_type = event_type

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None, known_evals=None):
        """
        Acts as a drop-in replacement for :func:`pybamm.Symbol.evaluate`
        """
        return self._expression.evaluate(t, y, y_dot, inputs, known_evals)

    def __str__(self):
        return self._name

    @property
    def name(self):
        return self._name

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value

    @property
    def event_type(self):
        return self._event_type
