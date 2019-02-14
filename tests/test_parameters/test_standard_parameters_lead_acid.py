#
# Test for the standard lead acid parameters
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from tests import shared

import unittest


class TestStandardParametersLeadAcid(unittest.TestCase):
    def test_parameters_defaults_lead_acid(self):
        # Load parameters to be tested
        parameters = {
            "Cd": pybamm.standard_parameters_lead_acid.Cd,
            "Crate": pybamm.standard_parameters_lead_acid.Crate,
            "iota_s_n": pybamm.standard_parameters_lead_acid.iota_s_n,
            "iota_s_p": pybamm.standard_parameters_lead_acid.iota_s_p,
            "gamma_dl_n": pybamm.standard_parameters_lead_acid.gamma_dl_n,
            "gamma_dl_p": pybamm.standard_parameters_lead_acid.gamma_dl_p,
            "DeltaVsurf_n": pybamm.standard_parameters_lead_acid.DeltaVsurf_n,
            "DeltaVsurf_p": pybamm.standard_parameters_lead_acid.DeltaVsurf_p,
            "alpha": pybamm.standard_parameters_lead_acid.alpha,
        }
        # Process
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"current scale": 1}
        )
        param_eval = {
            name: parameter_values.process_symbol(parameter).evaluate()
            for name, parameter in parameters.items()
        }

        # Diffusional C-rate should be smaller than C-rate
        self.assertLess(param_eval["Cd"], param_eval["Crate"])

        # Dimensionless electrode conductivities should be large
        self.assertGreater(param_eval["iota_s_n"], 10)
        self.assertGreater(param_eval["iota_s_p"], 10)
        # Dimensionless double-layer capacity should be small
        self.assertLess(param_eval["gamma_dl_n"], 1e-3)
        self.assertLess(param_eval["gamma_dl_p"], 1e-3)
        # Volume change positive in negative electrode and negative in positive
        # electrode
        self.assertLess(param_eval["DeltaVsurf_n"], 0)
        self.assertGreater(param_eval["DeltaVsurf_p"], 0)
        # Excluded volume fraction should be less than 0.1
        self.assertLess(abs(param_eval["alpha"]), 1e-1)

    def test_concatenated_parameters(self):
        # create
        s = pybamm.standard_parameters_lead_acid.s
        self.assertIsInstance(s, pybamm.Concatenation)
        self.assertEqual(
            s.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # process parameters and discretise
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"current scale": 1}
        )
        mesh = shared.MeshForTesting()
        disc = pybamm.BaseDiscretisation(mesh)
        processed_s = disc.process_symbol(parameter_values.process_symbol(s))

        # test output
        self.assertIsInstance(processed_s, pybamm.Vector)
        self.assertEqual(processed_s.shape, mesh["whole cell"].nodes.shape)

    @unittest.skip("lead acid functions not yet implemented")
    def test_functions_lead_acid(self):
        # Tests on how the parameters interact
        param = pybamm.ParameterValues(chemistry="lead-acid")
        mesh = pybamm.Mesh(param, 10)
        param.set_mesh(mesh)
        # Known values for dimensionless functions
        self.assertEqual(param.D_eff(1, 1), 1)
        self.assertEqual(param.kappa_eff(0, 1), 0)
        self.assertEqual(param.kappa_eff(1, 0), 0)
        self.assertEqual(param.neg_reactions.j0(0), 0)
        self.assertEqual(param.pos_reactions.j0(0), 0)
        # Known monotonicity for dimensionless functions
        self.assertLess(param.neg_reactions.j0(1), param.neg_reactions.j0(2))
        self.assertLess(param.pos_reactions.j0(1), param.pos_reactions.j0(2))
        self.assertGreater(param.lead_acid_misc.chi(1), param.lead_acid_misc.chi(0.5))
        self.assertLess(param.neg_reactions.U(1), param.neg_reactions.U(0.5))
        self.assertGreater(param.pos_reactions.U(1), param.pos_reactions.U(0.5))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
