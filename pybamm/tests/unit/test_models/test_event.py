#
# Tests Event class
#
from tests import TestCase
import pybamm
import numpy as np
import unittest


class TestEvent(TestCase):
    def test_event(self):
        expression = pybamm.Scalar(1)
        event = pybamm.Event("my event", expression)

        self.assertEqual(event.name, "my event")
        self.assertEqual(event.__str__(), "my event")
        self.assertEqual(event.expression, expression)
        self.assertEqual(event.event_type, pybamm.EventType.TERMINATION)

    def test_expression_evaluate(self):
        # Test t
        expression = pybamm.t
        event = pybamm.Event("my event", expression)
        self.assertEqual(event.evaluate(t=1), 1)

        # Test y
        sv = pybamm.StateVector(slice(0, 10))
        expression = sv
        eval_array = np.linspace(0, 2, 19)
        test_array = np.linspace(0, 1, 10)[:, np.newaxis]
        event = pybamm.Event("my event", expression)
        np.testing.assert_array_equal(event.evaluate(y=eval_array), test_array)

        # Test y_dot
        expression = sv.diff(pybamm.t)
        event = pybamm.Event("my event", expression)
        np.testing.assert_array_equal(event.evaluate(y_dot=eval_array), test_array)

    def test_event_types(self):
        event_types = [
            pybamm.EventType.TERMINATION,
            pybamm.EventType.DISCONTINUITY,
            pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            pybamm.EventType.SWITCH,
        ]

        for event_type in event_types:
            event = pybamm.Event("my event", pybamm.Scalar(1), event_type)
            self.assertEqual(event.event_type, event_type)

    def test_to_from_json(self):
        expression = pybamm.Scalar(1)
        event = pybamm.Event("my event", expression)

        event_json = {
            "name": "my event",
            "event_type": ["EventType.TERMINATION", 0],
        }

        event_ser_json = event.to_json()
        self.assertEqual(event_ser_json, event_json)

        event_json["expression"] = expression

        new_event = pybamm.Event._from_json(event_json)

        # check for equal expressions
        self.assertEqual(new_event.expression, event.expression)

        # check for equal event types
        self.assertEqual(new_event.event_type, event.event_type)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
