#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.LOQS()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        model = pybamm.lead_acid.LOQS({"convection": True})
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_default_spatial_methods(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_incompatible_options(self):
        options = {"bc_options": {"dimensionality": 1}}
        with self.assertRaises(pybamm.ModelError):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSWithSideReactions(unittest.TestCase):
    def test_well_posed(self):
        options = {"capacitance": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_varying_surface_area(self):
        options = {
            "capacitance": "differential",
            "side reactions": ["oxygen"],
            "interfacial surface area": "varying",
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_incompatible_options(self):
        options = {"side reactions": ["something"]}
        with self.assertRaises(pybamm.ModelError):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSCapacitance(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1D(self):
        options = {"capacitance": "differential", "bc_options": {"dimensionality": 1}}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    @unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
    def test_default_solver(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScipySolver)
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_default_geometry(self):
        options = {"capacitance": "differential"}
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
    unittest.main()
