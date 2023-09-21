#
# Tests for the InputParameter class
#
from tests import TestCase
import numpy as np
import pybamm
import unittest

import unittest.mock as mock


class TestInputParameter(TestCase):
    def test_input_parameter_init(self):
        a = pybamm.InputParameter("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.evaluate(inputs={"a": 1}), 1)
        self.assertEqual(a.evaluate(inputs={"a": 5}), 5)

        a = pybamm.InputParameter("a", expected_size=10)
        self.assertEqual(a._expected_size, 10)
        np.testing.assert_array_equal(
            a.evaluate(inputs="shape test"), np.nan * np.ones((10, 1))
        )
        y = np.linspace(0, 1, 10)
        np.testing.assert_array_equal(a.evaluate(inputs={"a": y}), y[:, np.newaxis])

        with self.assertRaisesRegex(
            ValueError,
            "Input parameter 'a' was given an object of size '1' but was expecting an "
            "object of size '10'",
        ):
            a.evaluate(inputs={"a": 5})

    def test_evaluate_for_shape(self):
        a = pybamm.InputParameter("a")
        self.assertTrue(np.isnan(a.evaluate_for_shape()))
        self.assertEqual(a.shape, ())

        a = pybamm.InputParameter("a", expected_size=10)
        self.assertEqual(a.shape, (10, 1))
        np.testing.assert_equal(a.evaluate_for_shape(), np.nan * np.ones((10, 1)))
        self.assertEqual(a.evaluate_for_shape().shape, (10, 1))

    def test_errors(self):
        a = pybamm.InputParameter("a")
        with self.assertRaises(TypeError):
            a.evaluate(inputs="not a dictionary")
        with self.assertRaises(KeyError):
            a.evaluate(inputs={"bad param": 5})
        # if u is not provided it gets turned into a dictionary and then raises KeyError
        with self.assertRaises(KeyError):
            a.evaluate()

    def test_to_from_json(self):
        a = pybamm.InputParameter("a")

        json_dict = {
            "name": "a",
            "id": mock.ANY,
            "domain": [],
            "expected_size": 1,
        }

        # to_json
        self.assertEqual(a.to_json(), json_dict)

        # from_json
        self.assertEqual(pybamm.InputParameter._from_json(json_dict), a)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
