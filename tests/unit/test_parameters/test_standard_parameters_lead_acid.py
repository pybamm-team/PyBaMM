#
# Test for the standard lead acid parameters
#
import pybamm
from tests import get_discretisation_for_testing

import unittest


class TestStandardParametersLeadAcid(unittest.TestCase):
    def test_scipy_constants(self):
        R = pybamm.standard_parameters_lead_acid.R
        self.assertAlmostEqual(R.evaluate(), 8.314, places=3)
        F = pybamm.standard_parameters_lead_acid.F
        self.assertAlmostEqual(F.evaluate(), 96485, places=0)

    def test_all_defined(self):
        parameters = pybamm.standard_parameters_lead_acid
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        output_file = "lead_acid_parameters.txt"
        pybamm.print_parameters(parameters, parameter_values, output_file)
        # test print_parameters with dict and without C-rate
        del parameter_values["Cell capacity [A.h]"]
        parameters = {
            "C_e": pybamm.standard_parameters_lead_acid.C_e,
            "sigma_n": pybamm.standard_parameters_lead_acid.sigma_n,
        }
        pybamm.print_parameters(parameters, parameter_values)

    def test_parameters_defaults_lead_acid(self):
        # Load parameters to be tested
        parameters = pybamm.standard_parameters_lead_acid
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        param_eval = pybamm.print_parameters(parameters, parameter_values)
        param_eval = {k: v[0] for k, v in param_eval.items()}

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
        s_param = pybamm.standard_parameters_lead_acid.s_plus_S
        self.assertIsInstance(s_param, pybamm.Concatenation)
        self.assertEqual(
            s_param.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # process parameters and discretise
        parameter_values = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Sulzer2019
        )
        disc = get_discretisation_for_testing()
        processed_s = disc.process_symbol(parameter_values.process_symbol(s_param))

        # test output
        combined_submeshes = disc.mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        self.assertEqual(processed_s.shape, (combined_submeshes[0].npts, 1))

    def test_current_functions(self):
        # create current functions
        dimensional_current_density = (
            pybamm.standard_parameters_lead_acid.dimensional_current_density_with_time
        )
        dimensionless_current_density = (
            pybamm.standard_parameters_lead_acid.current_with_time
        )

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode width [m]": 0.1,
                "Negative electrode thickness [m]": 1,
                "Separator thickness [m]": 1,
                "Positive electrode thickness [m]": 1,
                "Typical electrolyte concentration [mol.m-3]": 1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Typical current [A]": 2,
                "Current function [A]": 2,
            }
        )
        dimensional_current_density_eval = parameter_values.process_symbol(
            dimensional_current_density
        )
        dimensionless_current_density_eval = parameter_values.process_symbol(
            dimensionless_current_density
        )
        self.assertAlmostEqual(
            dimensional_current_density_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1)
        )
        self.assertEqual(dimensionless_current_density_eval.evaluate(t=3), 1)

    def test_functions_lead_acid(self):
        # Load parameters to be tested
        parameters = {
            "D_e_1": pybamm.standard_parameters_lead_acid.D_e(
                pybamm.Scalar(1), pybamm.Scalar(0)
            ),
            "kappa_e_0": pybamm.standard_parameters_lead_acid.kappa_e(
                pybamm.Scalar(0), pybamm.Scalar(0)
            ),
            "chi_1": pybamm.standard_parameters_lead_acid.chi(pybamm.Scalar(1)),
            "chi_0.5": pybamm.standard_parameters_lead_acid.chi(pybamm.Scalar(0.5)),
            "U_n_1": pybamm.standard_parameters_lead_acid.U_n(
                pybamm.Scalar(1), pybamm.Scalar(0)
            ),
            "U_n_0.5": pybamm.standard_parameters_lead_acid.U_n(
                pybamm.Scalar(0.5), pybamm.Scalar(0)
            ),
            "U_p_1": pybamm.standard_parameters_lead_acid.U_p(
                pybamm.Scalar(1), pybamm.Scalar(0)
            ),
            "U_p_0.5": pybamm.standard_parameters_lead_acid.U_p(
                pybamm.Scalar(0.5), pybamm.Scalar(0)
            ),
        }
        # Process
        parameter_values = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Sulzer2019
        )
        param_eval = pybamm.print_parameters(parameters, parameter_values)
        param_eval = {k: v[0] for k, v in param_eval.items()}

        # Known values for dimensionless functions
        self.assertEqual(param_eval["D_e_1"], 1)
        self.assertEqual(param_eval["kappa_e_0"], 0)
        # Known monotonicity for dimensionless functions
        self.assertGreater(param_eval["chi_1"], param_eval["chi_0.5"])
        self.assertLess(param_eval["U_n_1"], param_eval["U_n_0.5"])
        self.assertGreater(param_eval["U_p_1"], param_eval["U_p_0.5"])

    def test_update_initial_state_of_charge(self):
        # Load parameters to be tested
        parameters = pybamm.standard_parameters_lead_acid
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        param_eval = pybamm.print_parameters(parameters, parameter_values)
        param_eval = {k: v[0] for k, v in param_eval.items()}

        # Update initial state of charge
        parameter_values.update({"Initial State of Charge": 0.2})
        param_eval_update = pybamm.print_parameters(parameters, parameter_values)
        param_eval_update = {k: v[0] for k, v in param_eval_update.items()}

        # Test that relevant parameters have changed as expected
        self.assertLess(param_eval_update["q_init"], param_eval["q_init"])
        self.assertLess(param_eval_update["c_e_init"], param_eval["c_e_init"])
        self.assertLess(
            param_eval_update["epsilon_n_init"], param_eval["epsilon_n_init"]
        )
        self.assertEqual(
            param_eval_update["epsilon_s_init"], param_eval["epsilon_s_init"]
        )
        self.assertLess(
            param_eval_update["epsilon_p_init"], param_eval["epsilon_p_init"]
        )
        self.assertGreater(
            param_eval_update["curlyU_n_init"], param_eval["curlyU_n_init"]
        )
        self.assertGreater(
            param_eval_update["curlyU_p_init"], param_eval["curlyU_p_init"]
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
