#
# Tests for the State Vector class
#
from tests import TestCase
import pybamm
import numpy as np

import unittest
import unittest.mock as mock


class TestStateVector(TestCase):
    def test_evaluate(self):
        sv = pybamm.StateVector(slice(0, 10))
        y = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(
            sv.evaluate(y=y), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        # Try evaluating with a y that is too short
        y2 = np.ones(5)
        with self.assertRaisesRegex(
            ValueError, "y is too short, so value with slice is smaller than expected"
        ):
            sv.evaluate(y=y2)

    def test_evaluate_list(self):
        sv = pybamm.StateVector(slice(0, 11), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_array_almost_equal(
            sv.evaluate(y=y),
            np.concatenate([np.linspace(0, 1, 11), np.linspace(2, 3, 11)])[
                :, np.newaxis
            ],
        )
        sv = pybamm.StateVector(slice(0, 11), slice(11, 20), slice(20, 31))
        y = np.linspace(0, 3, 31)
        np.testing.assert_array_almost_equal(sv.evaluate(y=y), y[:, np.newaxis])

    def test_name(self):
        sv = pybamm.StateVector(slice(0, 10))
        self.assertEqual(sv.name, "y[0:10]")
        sv = pybamm.StateVector(slice(0, 10), slice(20, 30))
        self.assertEqual(sv.name, "y[0:10,20:30]")
        sv = pybamm.StateVector(
            slice(0, 10), slice(20, 30), slice(40, 50), slice(60, 70)
        )
        self.assertEqual(sv.name, "y[0:10,20:30,...,60:70]")

    def test_pass_evaluation_array(self):
        # Turn off debug mode for this test
        original_debug_mode = pybamm.settings.debug_mode
        pybamm.settings.debug_mode = False
        # Test that evaluation array gets passed down (doesn't have to be the correct
        # array for this test)
        array = np.array([1, 2, 3, 4, 5])
        sv = pybamm.StateVector(slice(0, 10), evaluation_array=array)
        np.testing.assert_array_equal(sv.evaluation_array, array)
        # Turn debug mode back to what is was before
        pybamm.settings.debug_mode = original_debug_mode

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "all y_slices must be slice objects"):
            pybamm.StateVector(slice(0, 10), 1)

    def test_to_from_json(self):
        original_debug_mode = pybamm.settings.debug_mode
        pybamm.settings.debug_mode = False

        array = np.array([1, 2, 3, 4, 5])
        sv = pybamm.StateVector(slice(0, 10), evaluation_array=array)

        json_dict = {
            "name": "y[0:10]",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "y_slice": [
                {
                    "start": 0,
                    "stop": 10,
                    "step": None,
                }
            ],
            "evaluation_array": [1, 2, 3, 4, 5],
        }

        self.assertEqual(sv.to_json(), json_dict)

        self.assertEqual(pybamm.StateVector._from_json(json_dict), sv)

        # Turn debug mode back to what is was before
        pybamm.settings.debug_mode = original_debug_mode


class TestStateVectorDot(TestCase):
    def test_evaluate(self):
        sv = pybamm.StateVectorDot(slice(0, 10))
        y_dot = np.linspace(0, 2, 19)
        np.testing.assert_array_equal(
            sv.evaluate(y_dot=y_dot), np.linspace(0, 1, 10)[:, np.newaxis]
        )

        # Try evaluating with a y that is too short
        y_dot2 = np.ones(5)
        with self.assertRaisesRegex(
            ValueError,
            "y_dot is too short, so value with slice is smaller than expected",
        ):
            sv.evaluate(y_dot=y_dot2)

        # Try evaluating with y_dot=None
        with self.assertRaisesRegex(
            TypeError,
            "StateVectorDot cannot evaluate input 'y_dot=None'",
        ):
            sv.evaluate(y_dot=None)

    def test_name(self):
        sv = pybamm.StateVectorDot(slice(0, 10))
        self.assertEqual(sv.name, "y_dot[0:10]")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
