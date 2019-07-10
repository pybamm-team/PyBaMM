#
# Tests for the lithium-ion SPM model
#
import pybamm
import unittest


class TestSPM(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPM(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" in model.default_geometry)
        self.assertNotIn("current collector", model.default_geometry)

        # Shape tests needs debugging for 2plus1D -- model works ok, but shape
        # test not right
        pybamm.settings.debug_mode = False
        options = {"bc_options": {"dimensionality": 2}}
        model = pybamm.lithium_ion.SPM(options)
        self.assertIn("current collector", model.default_geometry)
        pybamm.settings.debug_mode = True

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_well_posed_2plus1D(self):
        # Shape tests needs debugging for 2plus1D -- model works ok, but shape
        # test not right
        pybamm.settings.debug_mode = False
        options = {"bc_options": {"dimensionality": 2}}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()
        pybamm.settings.debug_mode = True

        options = {"bc_options": {"dimensionality": 1}}
        with self.assertRaises(NotImplementedError):
            model = pybamm.lithium_ion.SPM(options)

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"thermal": None}
        model = pybamm.lithium_ion.SPM(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        # Shape tests needs debugging for 2plus1D -- model works ok, but shape
        # test not right
        pybamm.settings.debug_mode = False
        options = {"bc_options": {"dimensionality": 2}}
        model = pybamm.lithium_ion.SPM(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)
        pybamm.settings.debug_mode = True


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
