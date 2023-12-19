import pybamm
import unittest
import tests
from tests import TestCase


class TestThevenin(TestCase):
    def test_basic_processing(self):
        model = pybamm.equivalent_circuit.Thevenin()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
