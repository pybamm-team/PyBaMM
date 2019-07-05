#
# Tests for the electrode-electrolyte interface equations for lithium-ion models
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
        del self.c_s_n_surf
        del self.c_s_p_surf

    def test_creation_lithium_ion(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.LithiumIonReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n, self.c_s_n_surf)
        j0_p = model.get_exchange_current_densities(self.c_e_p, self.c_s_p_surf)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_lithium_ion(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.LithiumIonReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n, self.c_s_n_surf)
        j0_p = model.get_exchange_current_densities(self.c_e_p, self.c_s_p_surf)
        # Process parameters
        parameter_values = model.default_parameter_values
        j0_n = parameter_values.process_symbol(j0_n)
        j0_p = parameter_values.process_symbol(j0_p)
        # Test
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_p.pre_order()]

    def test_discretisation_lithium_ion(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.LithiumIonReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n, self.c_s_n_surf)
        j0_p = model.get_exchange_current_densities(self.c_e_p, self.c_s_p_surf)
        # Process parameters and discretise
        parameter_values = model.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(
            [self.c_e, self.c_s_n_surf.orphans[0], self.c_s_p_surf.orphans[0]]
        )
        j0_n = disc.process_symbol(parameter_values.process_symbol(j0_n))
        j0_p = disc.process_symbol(parameter_values.process_symbol(j0_p))

        # Test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        submesh = mesh.combine_submeshes(*whole_cell)
        y = np.concatenate(
            [
                submesh[0].nodes ** 2,
                mesh["negative particle"][0].nodes,
                mesh["positive particle"][0].nodes,
            ]
        )
        # should evaluate to vectors with the right shape
        self.assertEqual(
            j0_n.evaluate(y=y).shape, (mesh["negative electrode"][0].npts, 1)
        )
        self.assertEqual(
            j0_p.evaluate(y=y).shape, (mesh["positive electrode"][0].npts, 1)
        )

    def test_failure(self):
        model = pybamm.interface.LithiumIonReaction(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_exchange_current_densities(None, None, "not a domain")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
