#
# Tests for the electrode-electrolyte interface equations
#
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestExchangeCurrentDensity(unittest.TestCase):
    def setUp(self):
        self.c_e_n = pybamm.Variable("concentration", domain=["negative electrode"])
        self.c_e_s = pybamm.Variable("concentration", domain=["separator"])
        self.c_e_p = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c_e = pybamm.Concatenation(self.c_e_n, self.c_e_s, self.c_e_p)
        self.c_s_n_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["negative particle"])
        )
        self.c_s_p_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["positive particle"])
        )

    def tearDown(self):
        del self.c_e_n
        del self.c_e_s
        del self.c_e_p
        del self.c_e

    def test_creation_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface_lead_acid.MainReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n)
        j0_p = model.get_exchange_current_densities(self.c_e_p)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface_lead_acid.MainReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n)
        j0_p = model.get_exchange_current_densities(self.c_e_p)
        # Process parameters
        parameter_values = model.default_parameter_values
        j0_n = parameter_values.process_symbol(j0_n)
        j0_p = parameter_values.process_symbol(j0_p)
        # Test
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_p.pre_order()]

    def test_discretisation_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface_lead_acid.MainReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n)
        j0_p = model.get_exchange_current_densities(self.c_e_p)
        # Process parameters and discretise
        parameter_values = model.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices([self.c_e])
        j0_n = disc.process_symbol(parameter_values.process_symbol(j0_n))
        j0_p = disc.process_symbol(parameter_values.process_symbol(j0_p))

        # Test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh.combine_submeshes(*whole_cell)
        y = submesh[0].nodes ** 2
        # should evaluate to vectors with the right shape
        self.assertEqual(
            j0_n.evaluate(y=y).shape, (mesh["negative electrode"][0].npts, 1)
        )
        self.assertEqual(
            j0_p.evaluate(y=y).shape, (mesh["positive electrode"][0].npts, 1)
        )

    def test_failure(self):
        model = pybamm.interface_lead_acid.MainReaction(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_exchange_current_densities(None, "not a domain")

    def test_diff_main_reaction(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface_lead_acid.MainReaction(param)
        parameter_values = model.default_parameter_values

        def j0_n(c_e):
            return model.get_exchange_current_densities(c_e, ["negative electrode"])

        def j0_p(c_e):
            return model.get_exchange_current_densities(c_e, ["positive electrode"])

        c_e = pybamm.Scalar(0.5)
        h = pybamm.Scalar(0.00001)

        # Analytical
        j0_n_diff = parameter_values.process_symbol(j0_n(c_e).diff(c_e))
        j0_p_diff = parameter_values.process_symbol(j0_p(c_e).diff(c_e))

        # Numerical
        j0_n_FD = parameter_values.process_symbol(
            (j0_n(c_e + h) - j0_n(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(j0_n_diff.evaluate(), j0_n_FD.evaluate())
        j0_p_FD = parameter_values.process_symbol(
            (j0_p(c_e + h) - j0_p(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(j0_p_diff.evaluate(), j0_p_FD.evaluate())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
