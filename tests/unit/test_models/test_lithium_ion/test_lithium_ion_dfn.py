#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest


class TestDFN(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lithium_ion.DFN(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lithium_ion.DFN(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("secondary" in model.default_geometry["negative particle"])

    def test_default_solver(self):
        options = {"thermal": None, "Voltage": "On"}
        model = pybamm.lithium_ion.DFN(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
