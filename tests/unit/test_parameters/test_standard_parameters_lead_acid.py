#
# Test for the standard lead acid parameters
#
import pybamm
from tests import get_discretisation_for_testing

import os
import unittest


class TestStandardParametersLeadAcid(unittest.TestCase):
    def test_scipy_constants(self):
        R = pybamm.standard_parameters_lead_acid.R
        self.assertAlmostEqual(R.evaluate(), 8.314, places=3)
        F = pybamm.standard_parameters_lead_acid.F
        self.assertAlmostEqual(F.evaluate(), 96485, places=0)

    def test_parameters_defaults_lead_acid(self):
        # Load parameters to be tested
        parameters = {
            "C_e": pybamm.standard_parameters_lead_acid.C_e,
            "C_rate": pybamm.standard_parameters_lead_acid.C_rate,
            "sigma_n": pybamm.standard_parameters_lead_acid.sigma_n,
            "sigma_p": pybamm.standard_parameters_lead_acid.sigma_p,
            "C_dl_n": pybamm.standard_parameters_lead_acid.C_dl_n,
            "C_dl_p": pybamm.standard_parameters_lead_acid.C_dl_p,
            "DeltaVsurf_n": pybamm.standard_parameters_lead_acid.DeltaVsurf_n,
            "DeltaVsurf_p": pybamm.standard_parameters_lead_acid.DeltaVsurf_p,
        }
        # Process
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current [A]": 1,
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Gu1997.py"
                ),
            },
        )
        param_eval = {
            name: parameter_values.process_symbol(parameter).evaluate()
            for name, parameter in parameters.items()
        }

        # Diffusional C-rate should be smaller than C-rate
        self.assertLess(param_eval["C_e"], param_eval["C_rate"])

        # Dimensionless electrode conductivities should be large
        self.assertGreater(param_eval["sigma_n"], 10)
        self.assertGreater(param_eval["sigma_p"], 10)
        # Dimensionless double-layer capacity should be small
        self.assertLess(param_eval["C_dl_n"], 1e-3)
        self.assertLess(param_eval["C_dl_p"], 1e-3)
        # Volume change positive in negative electrode and negative in positive
        # electrode
        self.assertLess(param_eval["DeltaVsurf_n"], 0)
        self.assertGreater(param_eval["DeltaVsurf_p"], 0)

    def test_concatenated_parameters(self):
        # create
        s_param = pybamm.standard_parameters_lead_acid.s
        self.assertIsInstance(s_param, pybamm.Concatenation)
        self.assertEqual(
            s_param.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # process parameters and discretise
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"Typical current [A]": 1}
        )
        disc = get_discretisation_for_testing()
        processed_s = disc.process_symbol(parameter_values.process_symbol(s_param))

        # test output
        combined_submeshes = disc.mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        self.assertEqual(processed_s.shape, combined_submeshes[0].nodes.shape)

    def test_current_functions(self):
        # create current functions
        dimensional_current = (
            pybamm.standard_parameters_lead_acid.dimensional_current_density_with_time
        )
        dimensionless_current = pybamm.standard_parameters_lead_acid.current_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height": 0.1,
                "Electrode depth": 0.1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Typical current [A]": 2,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            }
        )
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)
        dimensionless_current_eval = parameter_values.process_symbol(
            dimensionless_current
        )
        self.assertAlmostEqual(
            dimensional_current_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1)
        )
        self.assertEqual(dimensionless_current_eval.evaluate(t=3), 1)

    def test_functions_lead_acid(self):
        # Load parameters to be tested
        parameters = {
            "D_e_1": pybamm.standard_parameters_lead_acid.D_e(pybamm.Scalar(1)),
            "kappa_e_0": pybamm.standard_parameters_lead_acid.kappa_e(pybamm.Scalar(0)),
            "chi_1": pybamm.standard_parameters_lead_acid.chi(pybamm.Scalar(1)),
            "chi_0.5": pybamm.standard_parameters_lead_acid.chi(pybamm.Scalar(0.5)),
            "U_n_1": pybamm.standard_parameters_lead_acid.U_n(pybamm.Scalar(1)),
            "U_n_0.5": pybamm.standard_parameters_lead_acid.U_n(pybamm.Scalar(0.5)),
            "U_p_1": pybamm.standard_parameters_lead_acid.U_p(pybamm.Scalar(1)),
            "U_p_0.5": pybamm.standard_parameters_lead_acid.U_p(pybamm.Scalar(0.5)),
        }
        # Process
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current [A]": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Gu1997.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Gu1997.py"
                ),
                "Darken thermodynamic factor": os.path.join(
                    input_path, "darken_thermodynamic_factor_Chapman1968.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        param_eval = {
            name: parameter_values.process_symbol(parameter).evaluate()
            for name, parameter in parameters.items()
        }

        # Known values for dimensionless functions
        self.assertEqual(param_eval["D_e_1"], 1)
        self.assertEqual(param_eval["kappa_e_0"], 0)
        # Known monotonicity for dimensionless functions
        self.assertGreater(param_eval["chi_1"], param_eval["chi_0.5"])
        self.assertLess(param_eval["U_n_1"], param_eval["U_n_0.5"])
        self.assertGreater(param_eval["U_p_1"], param_eval["U_p_0.5"])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
