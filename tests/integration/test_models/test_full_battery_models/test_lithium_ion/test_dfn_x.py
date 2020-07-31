#
# Tests for the lithium-ion DFN model
#
import pybamm
import tests

import unittest


class TestDFN(unittest.TestCase):
    def test_particle_distribution_in_x(self):
        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values

        def negative_distribution(x):
            return 1 + x

        def positive_distribution(x):
            return 1 + (x - (1 - model.param.l_p))

        param["Negative particle distribution in x"] = negative_distribution
        param["Positive particle distribution in x"] = positive_distribution
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
