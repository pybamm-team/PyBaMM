#
# Tests for the electrolyte diffusion submodel
#
import pybamm
import unittest


class TestElectrolyteDiffusion(unittest.TestCase):
    def test_use_epsilon_param(self):
        parameters = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_diffusion.StefanMaxwell(parameters)
        c_e = pybamm.Variable("c")
        model.set_leading_order_system(c_e, {})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
