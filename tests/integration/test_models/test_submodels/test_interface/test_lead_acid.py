#
# Tests for the electrode-electrolyte interface equations for lead-acid models
#

import pybamm
from tests import get_discretisation_for_testing
import unittest


class TestMainReaction(unittest.TestCase):
    def setUp(self):
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration [mol.m-3]",
            domain=["negative electrode"],
        )
        c_e_s = pybamm.Variable(
            "Separator concentration [mol.m-3]", domain=["separator"]
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain=["positive electrode"],
        )
        self.c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)
        T_n = pybamm.Variable(
            "Negative electrode temperature [K]", domain=["negative electrode"]
        )
        T_s = pybamm.Variable("Separator temperature [K]", domain=["separator"])
        T_p = pybamm.Variable(
            "Positive electrode temperature [K]", domain=["positive electrode"]
        )
        self.T = pybamm.concatenation(T_n, T_s, T_p)
        self.variables = {
            "Negative electrolyte concentration [mol.m-3]": c_e_n,
            "Positive electrolyte concentration [mol.m-3]": c_e_p,
            "Negative electrode temperature [K]": T_n,
            "Positive electrode temperature [K]": T_p,
        }

    def tearDown(self):
        del self.variables
        del self.c_e

    def test_creation_main_reaction(self):
        # With intercalation
        param = pybamm.LeadAcidParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lead-acid main", {}
        )
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lead-acid main", {}
        )
        j0_p = model_p._get_exchange_current_density(self.variables)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_main_reaction(self):
        # With intercalation
        param = pybamm.LeadAcidParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lead-acid main", {}
        )
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lead-acid main", {}
        )
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
        param = pybamm.LeadAcidParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lead-acid main", {}
        )
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lead-acid main", {}
        )
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
        submesh = mesh[whole_cell]
        y = submesh.nodes**2
        # should evaluate to vectors with the right shape
        self.assertEqual(j0_n.evaluate(y=y).shape, (mesh["negative electrode"].npts, 1))
        self.assertEqual(j0_p.evaluate(y=y).shape, (mesh["positive electrode"].npts, 1))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
