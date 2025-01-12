#
# Tests for the Scalar class
#

import pybamm


class TestScalar:
    def test_scalar_eval(self):
        a = pybamm.Scalar(5)
        assert a.value == 5
        assert a.evaluate() == 5

    def test_scalar_operations(self):
        a = pybamm.Scalar(5)
        b = pybamm.Scalar(6)
        assert (a + b).evaluate() == 11
        assert (a - b).evaluate() == -1
        assert (a * b).evaluate() == 30
        assert (a / b).evaluate() == 5 / 6

    def test_scalar_eq(self):
        a1 = pybamm.Scalar(4)
        a2 = pybamm.Scalar(4)
        assert a1 == a2
        a3 = pybamm.Scalar(5)
        assert a1 != a3

    def test_to_equation(self):
        a = pybamm.Scalar(3)
        b = pybamm.Scalar(4)

        # Test value
        assert str(a.to_equation()) == "3.0"

        # Test print_name
        b.print_name = "test"
        assert str(b.to_equation()) == "test"

    def test_copy(self):
        a = pybamm.Scalar(5)
        b = a.create_copy()
        assert a == b

    def test_to_from_json(self, mocker):
        a = pybamm.Scalar(5)
        json_dict = {"name": "5.0", "id": mocker.ANY, "value": 5.0}

        assert a.to_json() == json_dict

        assert pybamm.Scalar._from_json(json_dict) == a
