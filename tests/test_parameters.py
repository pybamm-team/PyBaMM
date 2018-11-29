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

    def test_parameters_init(self):
        with self.assertRaises(NotImplementedError):
            pybamm.Parameters(tests="not a test")
        with self.assertRaises(NotImplementedError):
            param = pybamm.Parameters(chemistry="not a chemistry")
        for chemistry in ["lead-acid"]:  # pybamm.KNOWN_CHEMISTRIES:
            param = pybamm.Parameters(chemistry=chemistry)
            self.assertEqual(param._raw["R"], 8.314)
            self.assertEqual(param._func.D_eff(param, 1, 1), 1)

    def test_parameters_options(self):
        # test dictionary input
        param = pybamm.Parameters(
            chemistry="lead-acid",
            optional_parameters={"Ln": 1 / 3, "Ls": 0.25, "Lp": 0.25},
        )
        self.assertEqual(param._raw["Ln"], 1 / 3)
        self.assertEqual(param._raw["R"], 8.314)
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )

        # Test file input
        param = pybamm.Parameters(
            chemistry="lead-acid", optional_parameters="optional_test.csv"
        )
        self.assertEqual(param._raw["Ln"], 0.5)
        self.assertEqual(param._raw["R"], 8.314)
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )

    def test_parameters_update_raw(self):
        param = pybamm.Parameters(chemistry="lead-acid")
        param.update_raw({"Ln": 0.5})
        self.assertEqual(param._raw["Ln"], 0.5)
        self.assertAlmostEqual(
            param.geometric["ln"] + param.geometric["ls"] + param.geometric["lp"],
            1,
            places=10,
        )

    def test_parameters_defaults_lead_acid(self):
        # Tests on how the parameters interact
        param = pybamm.Parameters(chemistry="lead-acid")
        mesh = pybamm.Mesh(param, 10)
        param.set_mesh(mesh)

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
        # Volume change positive in negative electrode and negative in positive
        # electrode
        self.assertGreater(param.neg_volume_changes["DeltaVsurf"], 0)
        self.assertLess(param.pos_volume_changes["DeltaVsurf"], 0)
        # Excluded volume fraction should be less than 1
        self.assertLess(param.lead_acid_misc["pi_os"], 1e-4)

    def test_functions_lead_acid(self):
        # Tests on how the parameters interact
        param = pybamm.Parameters(chemistry="lead-acid")
        mesh = pybamm.Mesh(param, 10)
        param.set_mesh(mesh)
        # Known values for dimensionless functions
        self.assertEqual(param.electrolyte["D_eff"](1, 1), 1)
        # Known monotonicity for dimensionless functions
        self.assertGreater(
            param.lead_acid_misc["chi"](1), param.lead_acid_misc["chi"](0.5)
        )
        self.assertLess(param.neg_reactions["U"](1), param.neg_reactions["U"](0.5))
        self.assertGreater(param.pos_reactions["U"](1), param.pos_reactions["U"](0.5))

    def test_mesh_dependent_parameters(self):
        param = pybamm.Parameters(chemistry="lead-acid")
        mesh = pybamm.Mesh(param, 10)
        param.set_mesh(mesh)

        self.assertEqual(param.electrolyte["s"].shape, mesh.xc.shape)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
