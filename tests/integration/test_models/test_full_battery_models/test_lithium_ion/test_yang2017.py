import pybamm
import unittest
import tests
from tests import TestCase


class TestYang2017(TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.Yang2017()
        parameter_values = pybamm.ParameterValues("OKane2022")
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
