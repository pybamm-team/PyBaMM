#
# Tests for the electrode-electrolyte interface equations for lead-acid models
#
import pybamm
from tests import get_discretisation_for_testing

import unittest


class TestMainReaction(unittest.TestCase):
    def setUp(self):
        c_e_n = pybamm.Variable("concentration", domain=["negative electrode"])
        c_e_s = pybamm.Variable("concentration", domain=["separator"])
        c_e_p = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        T_n = pybamm.Variable("temperature", domain=["negative electrode"])
        T_s = pybamm.Variable("temperature", domain=["separator"])
        T_p = pybamm.Variable("temperature", domain=["positive electrode"])
        self.T = pybamm.Concatenation(T_n, T_s, T_p)
        self.variables = {
            "Negative electrolyte concentration": c_e_n,
            "Positive electrolyte concentration": c_e_p,
            "Negative electrode temperature": T_n,
            "Positive electrode temperature": T_p,
        }

    def tearDown(self):
        del self.variables
        del self.c_e

    def test_creation_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.BaseInterface(param, "Negative", "lead-acid main")
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(param, "Positive", "lead-acid main")
        j0_p = model_p._get_exchange_current_density(self.variables)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.BaseInterface(param, "Negative", "lead-acid main")
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(param, "Positive", "lead-acid main")
        j0_p = model_p._get_exchange_current_density(self.variables)
        # Process parameters
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        j0_n = parameter_values.process_symbol(j0_n)
        j0_p = parameter_values.process_symbol(j0_p)
        # Test
        for x in j0_n.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)
        for x in j0_p.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)

    def test_discretisation_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.BaseInterface(param, "Negative", "lead-acid main")
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(param, "Positive", "lead-acid main")
        j0_p = model_p._get_exchange_current_density(self.variables)
        # Process parameters and discretise
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices([self.c_e])
        j0_n = disc.process_symbol(parameter_values.process_symbol(j0_n))
        j0_p = disc.process_symbol(parameter_values.process_symbol(j0_p))

        # Test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh.combine_submeshes(*whole_cell)
        y = submesh.nodes ** 2
        # should evaluate to vectors with the right shape
        self.assertEqual(j0_n.evaluate(y=y).shape, (mesh["negative electrode"].npts, 1))
        self.assertEqual(j0_p.evaluate(y=y).shape, (mesh["positive electrode"].npts, 1))

    def test_diff_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.BaseInterface(param, "Negative", "lead-acid main")
        model_p = pybamm.interface.BaseInterface(param, "Positive", "lead-acid main")
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values

        def j0_n(c_e):
            variables = {**self.variables, "Negative electrolyte concentration": c_e}
            return model_n._get_exchange_current_density(variables)

        def j0_p(c_e):
            variables = {**self.variables, "Positive electrolyte concentration": c_e}
            return model_p._get_exchange_current_density(variables)

        c_e = pybamm.InputParameter("c_e")
        h = pybamm.Scalar(0.00001)

        # Analytical
        j0_n_diff = parameter_values.process_symbol(j0_n(c_e).diff(c_e))
        j0_p_diff = parameter_values.process_symbol(j0_p(c_e).diff(c_e))

        # Numerical
        j0_n_FD = parameter_values.process_symbol(
            (j0_n(c_e + h) - j0_n(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(
            j0_n_diff.evaluate(inputs={"c_e": 0.5}),
            j0_n_FD.evaluate(inputs={"c_e": 0.5}),
        )
        j0_p_FD = parameter_values.process_symbol(
            (j0_p(c_e + h) - j0_p(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(
            j0_p_diff.evaluate(inputs={"c_e": 0.5}),
            j0_p_FD.evaluate(inputs={"c_e": 0.5}),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
