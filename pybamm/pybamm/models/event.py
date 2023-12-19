from enum import Enum


class EventType(Enum):
    """
    Defines the type of event, see :class:`pybamm.Event`

    TERMINATION indicates an event that will terminate the solver, the expression should
    return 0 when the event is triggered

    DISCONTINUITY indicates an expected discontinuity in the solution, the expression
    should return the time that the discontinuity occurs. The solver will integrate up
    to the discontinuity and then restart just after the discontinuity.

    INTERPOLANT_EXTRAPOLATION indicates that a pybamm.Interpolant object has been
    evaluated outside of the range.

    SWITCH indicates an event switch that is used in CasADI "fast with events" model.
    """

    TERMINATION = 0
    DISCONTINUITY = 1
    INTERPOLANT_EXTRAPOLATION = 2
    SWITCH = 3


class Event:
    """

    Defines an event for use within a pybamm model

    Attributes
    ----------

    name: str
        A string giving the name of the event.
    expression: :class:`pybamm.Symbol`
        An expression that defines when the event occurs.
    event_type: :class:`pybamm.EventType` (optional)
        An enum defining the type of event. By default it is set to TERMINATION.

    """

    def __init__(self, name, expression, event_type=EventType.TERMINATION):
        self._name = name
        self._expression = expression
        self._event_type = event_type

    @classmethod
    def _from_json(cls, snippet: dict):
        """
        Reconstructs an Event instance during deserialisation of a JSON file.

        Parameters
        ----------
        snippet: dict
            Contains the information needed to reconstruct a specific instance.
            Should contain "name", "expression" and "event_type".
        """

        instance = cls.__new__(cls)

        instance.__init__(
            snippet["name"],
            snippet["expression"],
            event_type=EventType(snippet["event_type"][1]),
        )

        return instance

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """
        Acts as a drop-in replacement for :func:`pybamm.Symbol.evaluate`
        """
        return self._expression.evaluate(t, y, y_dot, inputs)

    def __str__(self):
        return self._name

    @property
    def name(self):
        return self._name

    @property
    def expression(self):
        return self._expression

    @property
    def event_type(self):
        return self._event_type

    def to_json(self):
        """
        Method to serialise an Event object into JSON.

        The expression is written out seperately,
        See :meth:`pybamm.Serialise._SymbolEncoder.default()`
        """

        # event_type contains string name, for JSON readability,
        # and value for deserialisation.

        json_dict = {
            "name": self._name,
            "event_type": [str(self._event_type), self._event_type.value],
        }

        return json_dict
