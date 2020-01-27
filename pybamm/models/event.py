import pybamm

from enum import Enum


class EventType(Enum):
    """Defines the type of event, see Event"""
    TERMINATION = 0
    DISCONTINUITY = 1


class Event:
    """

    Defines an event for use within a pybamm model

    Attributes
    ----------

    name: str
        A string giving the name of the event
    event_type: EventType
        An enum defining the type of event, see EventType
    expression: pybamm.Symbol
        An expression that defines when the event occurs


    """

    def __init__(self, name, expression, event_type=EventType.TERMINATION):
        self._name = name
        self._expression = expression
        self._event_type = event_type

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
        return event_type
