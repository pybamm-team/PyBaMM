#
# Tests for the Effective Current Collector Resistance model
#
import pybamm
import unittest


class TestEffectiveResistance2D(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("current collector" in model.default_geometry)
        self.assertNotIn("negative electrode", model.default_geometry)

    def test_default_solver(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        self.assertIsInstance(model.default_solver, pybamm.AlgebraicSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
