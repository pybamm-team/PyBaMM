#
# Tests getting model info
#
from tests import TestCase
import pybamm
import unittest


class TestModelInfo(TestCase):
    def test_find_parameter_info(self):
        model = pybamm.lithium_ion.SPM()
        model.info("Negative electrode diffusivity [m2.s-1]")
        model = pybamm.lithium_ion.SPMe()
        model.info("Negative electrode diffusivity [m2.s-1]")
        model = pybamm.lithium_ion.DFN()
        model.info("Negative electrode diffusivity [m2.s-1]")

        model.info("Not a parameter")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
