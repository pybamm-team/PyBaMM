#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest


class TestDFN(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.DFN()
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.lithium_ion.DFN()
        self.assertTrue(isinstance(model.default_geometry, pybamm.Geometry))
        self.assertTrue("secondary" in model.default_geometry["negative particle"])

    def test_default_solver(self):
        model = pybamm.lithium_ion.DFN()
        self.assertTrue(isinstance(model.default_solver, pybamm.ScikitsDaeSolver))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
