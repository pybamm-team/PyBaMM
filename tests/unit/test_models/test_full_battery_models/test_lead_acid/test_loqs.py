#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"thermal": None, "convection": True}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_default_spatial_methods(self):
        options = {"thermal": None}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_incompatible_options(self):
        options = {"bc_options": {"dimensionality": 1}}
        with self.assertRaisesRegex(pybamm.OptionError, "must use surface formulation"):
            pybamm.lead_acid.LOQS(options)
        options = {"surface form": "bad surface form"}
        with self.assertRaisesRegex(pybamm.OptionError, "surface form"):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSWithSideReactions(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_varying_surface_area(self):
        options = {
            "surface form": "differential",
            "side reactions": ["oxygen"],
            "interfacial surface area": "varying",
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_incompatible_options(self):
        options = {"side reactions": ["something"]}
        with self.assertRaises(pybamm.OptionError):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSSurfaceForm(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1D(self):
        options = {"surface form": "differential", "bc_options": {"dimensionality": 1}}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"surface form": "differential", "bc_options": {"dimensionality": 1}}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_default_geometry(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertNotIn("current collector", model.default_geometry)
        options["bc_options"] = {"dimensionality": 1}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIn("current collector", model.default_geometry)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
