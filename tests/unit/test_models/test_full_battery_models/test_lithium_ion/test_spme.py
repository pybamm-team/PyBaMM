#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest


class TestSPMe(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" in model.default_geometry)

        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIn("current collector", model.default_geometry)

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIn("current collector", model.default_geometry)

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "single particle potential pair",
            "dimensionality": 2,
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"bc_options": {"dimensionality": 5}}
        with self.assertRaises(pybamm.OptionError):
            model = pybamm.lithium_ion.SPMe(options)

    def test_thermal(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"thermal": "full"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "full",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "full",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
