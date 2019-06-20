#
# Tests for the electrolyte diffusion submodel
#
import pybamm
import unittest


class TestElectrolyteDiffusion(unittest.TestCase):
    def test_use_epsilon_param(self):
        pybamm.standard_parameters_lithium_ion


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
