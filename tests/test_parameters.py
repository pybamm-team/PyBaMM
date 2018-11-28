#
# Test for the parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestParameters(unittest.TestCase):
    def test_read_parameters_csv(self):
        data = pybamm.read_parameters_csv("lead-acid", "default.csv")
        self.assertEqual(data["R"], 8.314)

    def test_parameters_defaults_lead_acid(self):
        # basic tests on how the parameters interact
        param = pybamm.Parameters()
        param.update_raw()
        self.assertEqual(param._raw["R"], 8.314)
        # Dimensionless lengths sum to 1
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )
        # Diffusional C-rate should be smaller than C-rate
        self.assertLess(param.electrolyte["Cd"], param.electrical["Crate"])
        # Dimensionless electrode conductivities should be large
        self.assertGreater(param.neg_electrode["iota_s"], 10)
        self.assertGreater(param.pos_electrode["iota_s"], 10)
        # Dimensionless double-layer capacity should be small
        self.assertLess(param.neg_reactions["gamma_dl"], 1e-3)
        self.assertLess(param.pos_reactions["gamma_dl"], 1e-3)
        # Volume change negative in negative electrode and positive in positive
        # electrode
        self.assertLess(param.neg_volume_changes["DeltaVsurf"], 0)
        self.assertGreater(param.pos_volume_changes["DeltaVsurf"], 0)
        # Excluded volume fraction should be less than 1
        self.assertLess(param.lead_acid_misc["alpha"], 1)

    def test_parameters_options(self):
        # Dictionary input
        param = pybamm.Parameters()
        param.update_raw(optional_parameters={"Ln": 1 / 3, "Ls": 0.25, "Lp": 0.25})
        self.assertEqual(param._raw["Ln"], 1 / 3)
        self.assertEqual(param._raw["R"], 8.314)
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )

        # File input
        param = pybamm.Parameters()
        param.update_raw(optional_parameters="optional_test.csv")
        self.assertEqual(param._raw["Ln"], 0.5)
        self.assertEqual(param._raw["R"], 8.314)
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )

    def test_parameters_tests(self):
        with self.assertRaises(NotImplementedError):
            pybamm.Parameters(tests="not a test")

    def test_mesh_dependent_parameters(self):
        param = pybamm.Parameters()
        param.update_raw()

        mesh = pybamm.Mesh(param, 10)
        param.set_mesh_dependent_parameters(mesh)
        self.assertEqual(param.s.shape, mesh.xc.shape)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
