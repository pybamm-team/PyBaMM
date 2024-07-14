#
# Tests for the InputParameter class
#
import numpy as np
import pybamm
import pytest
import unittest.mock as mock


class TestInputParameter:
    def test_input_parameter_init(self):
        a = pybamm.InputParameter("a")
        assert a.name == "a"
        assert a.evaluate(inputs={"a": 1}) == 1
        assert a.evaluate(inputs={"a": 5}) == 5

        a = pybamm.InputParameter("a", expected_size=10)
        assert a._expected_size == 10
        np.testing.assert_array_equal(
            a.evaluate(inputs="shape test"), np.nan * np.ones((10, 1))
        )
        y = np.linspace(0, 1, 10)
        np.testing.assert_array_equal(a.evaluate(inputs={"a": y}), y[:, np.newaxis])

        with pytest.raises(
            ValueError,
            match="Input parameter 'a' was given an object of size '1' but was expecting an "
            "object of size '10'",
        ):
            a.evaluate(inputs={"a": 5})

    def test_evaluate_for_shape(self):
        a = pybamm.InputParameter("a")
        assert np.isnan(a.evaluate_for_shape())
        assert a.shape == ()

        a = pybamm.InputParameter("a", expected_size=10)
        assert a.shape == (10, 1)
        np.testing.assert_equal(a.evaluate_for_shape(), np.nan * np.ones((10, 1)))
        assert a.evaluate_for_shape().shape == (10, 1)

    def test_errors(self):
        a = pybamm.InputParameter("a")
        with pytest.raises(TypeError):
            a.evaluate(inputs="not a dictionary")
        with pytest.raises(KeyError):
            a.evaluate(inputs={"bad param": 5})
        # if u is not provided it gets turned into a dictionary and then raises KeyError
        with pytest.raises(KeyError):
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
        assert a.to_json() == json_dict

        # from_json
        assert pybamm.InputParameter._from_json(json_dict) == a
