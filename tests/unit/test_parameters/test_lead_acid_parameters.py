#
# Test for the standard lead acid parameters
#
import pybamm
from tests import get_discretisation_for_testing

import unittest


class TestStandardParametersLeadAcid(unittest.TestCase):
    def test_scipy_constants(self):
        param = pybamm.LeadAcidParameters()
        self.assertAlmostEqual(param.R.evaluate(), 8.314, places=3)
        self.assertAlmostEqual(param.F.evaluate(), 96485, places=0)

    def test_print_parameters(self):
        parameters = pybamm.LeadAcidParameters()
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        output_file = "lead_acid_parameters.txt"
        parameter_values.print_parameters(parameters, output_file)

    def test_parameters_defaults_lead_acid(self):
        # Load parameters to be tested
        parameters = pybamm.LeadAcidParameters()
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        param_eval = parameter_values.print_parameters(parameters)

        # Volume change positive in negative electrode and negative in positive
        # electrode
        self.assertLess(param_eval["n.DeltaVsurf"], 0)
        self.assertGreater(param_eval["p.DeltaVsurf"], 0)

    def test_concatenated_parameters(self):
        # create
        param = pybamm.LeadAcidParameters()
        eps_param = param.epsilon_init
        self.assertIsInstance(eps_param, pybamm.Concatenation)
        self.assertEqual(
            eps_param.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # process parameters and discretise
        parameter_values = pybamm.ParameterValues("Sulzer2019")
        disc = get_discretisation_for_testing()
        processed_eps = disc.process_symbol(parameter_values.process_symbol(eps_param))

        # test output
        submeshes = disc.mesh[("negative electrode", "separator", "positive electrode")]
        self.assertEqual(processed_eps.shape, (submeshes.npts, 1))

    def test_current_functions(self):
        # create current functions
        param = pybamm.LeadAcidParameters()
        current_density = param.current_density_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height [m]": 0.1,
                "Electrode width [m]": 0.1,
                "Negative electrode thickness [m]": 1,
                "Separator thickness [m]": 1,
                "Positive electrode thickness [m]": 1,
                "Initial concentration in electrolyte [mol.m-3]": 1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Current function [A]": 2,
            }
        )
        current_density_eval = parameter_values.process_symbol(current_density)
        self.assertAlmostEqual(current_density_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1))

    def test_thermal_parameters(self):
        values = pybamm.lead_acid.BaseModel().default_parameter_values
        param = pybamm.LeadAcidParameters()
        T = 300  # dummy temperature as the values are constant

        # Density
        self.assertEqual(values.evaluate(param.n.rho_c_p_cc(T)), 11300 * 130)
        self.assertEqual(values.evaluate(param.n.rho_c_p(T)), 11300 * 130)
        self.assertEqual(values.evaluate(param.s.rho_c_p(T)), 1680 * 700)
        self.assertEqual(values.evaluate(param.p.rho_c_p(T)), 9375 * 256)
        self.assertEqual(values.evaluate(param.p.rho_c_p_cc(T)), 9375 * 256)

        # Thermal conductivity
        self.assertEqual(values.evaluate(param.n.lambda_cc(T)), 35)
        self.assertEqual(values.evaluate(param.n.lambda_(T)), 35)
        self.assertEqual(values.evaluate(param.s.lambda_(T)), 0.04)
        self.assertEqual(values.evaluate(param.p.lambda_(T)), 35)
        self.assertEqual(values.evaluate(param.p.lambda_cc(T)), 35)

    def test_functions_lead_acid(self):
        # Load parameters to be tested
        param = pybamm.LeadAcidParameters()
        T = 300
        c_e_1 = 1000
        c_e_pt5 = 500
        parameters = {
            "chi_1": param.chi(c_e_1, T),
            "chi_0.5": param.chi(c_e_pt5, T),
            "U_n_1": param.n.prim.U(c_e_1, T),
            "U_n_0.5": param.n.prim.U(c_e_pt5, T),
            "U_p_1": param.p.prim.U(c_e_1, T),
            "U_p_0.5": param.p.prim.U(c_e_pt5, T),
            "j0_Ox_1": param.p.prim.j0_Ox(c_e_1, T),
            "j0_Ox_0.5": param.p.prim.j0_Ox(c_e_pt5, T),
        }
        # Process
        parameter_values = pybamm.ParameterValues("Sulzer2019")
        param_eval = parameter_values.print_parameters(parameters)

        # Known monotonicity for functions
        self.assertGreater(param_eval["chi_1"], param_eval["chi_0.5"])
        self.assertLess(param_eval["U_n_1"], param_eval["U_n_0.5"])
        self.assertGreater(param_eval["U_p_1"], param_eval["U_p_0.5"])
        self.assertGreater(param_eval["j0_Ox_1"], param_eval["j0_Ox_0.5"])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
