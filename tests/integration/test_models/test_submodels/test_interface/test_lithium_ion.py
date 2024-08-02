#
# Tests for the electrode-electrolyte interface equations for lithium-ion models
#

import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestExchangeCurrentDensity(unittest.TestCase):
    def setUp(self):
        c_e_n = pybamm.Variable("concentration", domain=["negative electrode"])
        c_e_s = pybamm.Variable("concentration", domain=["separator"])
        c_e_p = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)
        self.c_s_n_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["negative particle"])
        )
        self.c_s_p_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["positive particle"])
        )
        self.variables = {
            "Negative electrolyte concentration [mol.m-3]": c_e_n,
            "Positive electrolyte concentration [mol.m-3]": c_e_p,
            "Negative particle surface concentration [mol.m-3]": self.c_s_n_surf,
            "Positive particle surface concentration [mol.m-3]": self.c_s_p_surf,
            "Negative electrode temperature [K]": 300,
            "Positive electrode temperature [K]": 300,
        }
        self.options = pybamm.BatteryModelOptions({"particle size": "single"})

    def tearDown(self):
        del self.variables
        del self.c_e
        del self.c_s_n_surf
        del self.c_s_p_surf

    def test_creation_lithium_ion(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lithium-ion main", {}
        )
        model_n.options = self.options
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lithium-ion main", {}
        )
        model_p.options = self.options
        j0_p = model_p._get_exchange_current_density(self.variables)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_lithium_ion(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lithium-ion main", {}
        )
        model_n.options = self.options
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lithium-ion main", {}
        )
        model_p.options = self.options
        j0_p = model_p._get_exchange_current_density(self.variables)
        # Process parameters
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values
        j0_n = parameter_values.process_symbol(j0_n)
        j0_p = parameter_values.process_symbol(j0_p)
        # Test
        for x in j0_n.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)
        for x in j0_p.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)

    def test_discretisation_lithium_ion(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.interface.BaseInterface(
            param, "negative", "lithium-ion main", {}
        )
        model_n.options = self.options
        j0_n = model_n._get_exchange_current_density(self.variables)
        model_p = pybamm.interface.BaseInterface(
            param, "positive", "lithium-ion main", {}
        )
        model_p.options = self.options
        j0_p = model_p._get_exchange_current_density(self.variables)
        # Process parameters and discretise
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(
            [self.c_e, self.c_s_n_surf.orphans[0], self.c_s_p_surf.orphans[0]]
        )
        j0_n = disc.process_symbol(parameter_values.process_symbol(j0_n))
        j0_p = disc.process_symbol(parameter_values.process_symbol(j0_p))

        # Test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh[whole_cell]
        y = np.concatenate(
            [
                submesh.nodes**2,
                mesh["negative particle"].nodes,
                mesh["positive particle"].nodes,
            ]
        )
        # should evaluate to vectors with the right shape
        self.assertEqual(j0_n.evaluate(y=y).shape, (mesh["negative electrode"].npts, 1))
        self.assertEqual(j0_p.evaluate(y=y).shape, (mesh["positive electrode"].npts, 1))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
