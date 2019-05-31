#
# Tests for the electrode-electrolyte interface equations
#
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestHomogeneousReaction(unittest.TestCase):
    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values

        j_n = parameter_values.process_symbol(j_n)
        j_p = parameter_values.process_symbol(j_p)

        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j_n.pre_order()])
        )
        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j_p.pre_order()])
        )
        self.assertEqual(j_n.domain, [])
        self.assertEqual(j_p.domain, [])

    def test_discretisation(self):
        disc = get_discretisation_for_testing()

        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values

        j_n = disc.process_symbol(parameter_values.process_symbol(j_n))
        j_p = disc.process_symbol(parameter_values.process_symbol(j_p))

        # test values
        l_n = parameter_values.process_symbol(param.l_n)
        l_p = parameter_values.process_symbol(param.l_p)
        np.testing.assert_array_equal((l_n * j_n).evaluate(0, None), 1)
        np.testing.assert_array_equal((l_p * j_p).evaluate(0, None), -1)

    def test_simplify_constant_current(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values
        j = pybamm.Concatenation(
            pybamm.Broadcast(j_n, ["negative electrode"]),
            pybamm.Broadcast(0, ["separator"]),
            pybamm.Broadcast(j_p, ["positive electrode"]),
        )

        j = disc.process_symbol(parameter_values.process_symbol(j))

        # Simplifiy, since current is constant this should give a vector
        j_simp = j.simplify()
        self.assertIsInstance(j_simp, pybamm.Vector)
        # test values
        l_n = parameter_values.process_symbol(param.l_n)
        l_p = parameter_values.process_symbol(param.l_p)
        npts_n = mesh["negative electrode"][0].npts
        npts_s = mesh["separator"][0].npts
        np.testing.assert_array_equal((l_n * j_simp).evaluate(0, None)[:npts_n], 1)
        np.testing.assert_array_equal(
            j_simp.evaluate(0, None)[npts_n : npts_n + npts_s], 0
        )
        np.testing.assert_array_equal(
            (l_p * j_simp).evaluate(0, None)[npts_n + npts_s :], -1
        )

    def test_failure(self):
        model = pybamm.interface.InterfacialCurrent(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_homogeneous_interfacial_current(None, "not a domain")


class TestButlerVolmer(unittest.TestCase):
    def setUp(self):
        self.eta_r_n = pybamm.Variable("overpotential", ["negative electrode"])
        self.eta_r_p = pybamm.Variable("overpotential", ["positive electrode"])
        self.j0_n = pybamm.Variable("exchange-curr density", ["negative electrode"])
        self.j0_p = pybamm.Variable("exchange-curr density", ["positive electrode"])

    def tearDown(self):
        del self.eta_r_n
        del self.eta_r_p
        del self.j0_n
        del self.j0_p

    def test_creation(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_butler_volmer(self.j0_n, self.eta_r_n)
        j_p = model.get_butler_volmer(self.j0_p, self.eta_r_p)

        # negative electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_n, pybamm.Multiplication)
        self.assertEqual(j_n.domain, ["negative electrode"])

        # positive electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_p, pybamm.Multiplication)
        self.assertEqual(j_p.domain, ["positive electrode"])

    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_butler_volmer(self.j0_n, self.eta_r_n)
        j_p = model.get_butler_volmer(self.j0_p, self.eta_r_p)

        # Process parameters
        parameter_values = model.default_parameter_values
        j_n = parameter_values.process_symbol(j_n)
        j_p = parameter_values.process_symbol(j_p)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j_p.pre_order()]

    def test_discretisation(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_n = model.get_butler_volmer(self.j0_n, self.eta_r_n)
        j_p = model.get_butler_volmer(self.j0_p, self.eta_r_p)
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)

        # Process parameters and discretise
        parameter_values = model.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices([self.j0_n, self.eta_r_n, self.j0_p, self.eta_r_p])

        j_n = disc.process_symbol(parameter_values.process_symbol(j_n))
        j_p = disc.process_symbol(parameter_values.process_symbol(j_p))
        j = disc.process_symbol(parameter_values.process_symbol(j))

        # test butler-volmer in each electrode
        submesh = np.concatenate(
            [mesh["negative electrode"][0].nodes, mesh["positive electrode"][0].nodes]
        )
        y = np.concatenate([submesh ** 2, submesh ** 3])
        self.assertEqual(
            j_n.evaluate(None, y).shape, (mesh["negative electrode"][0].npts, 1)
        )
        self.assertEqual(
            j_p.evaluate(None, y).shape, (mesh["positive electrode"][0].npts, 1)
        )

        # test concatenated butlver-volmer
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh.combine_submeshes(*whole_cell)
        self.assertEqual(j.evaluate(None, y).shape, (whole_cell_mesh[0].npts, 1))

    def test_failure(self):
        model = pybamm.interface.InterfacialCurrent(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_butler_volmer(None, None, "not a domain")
        with self.assertRaises(pybamm.DomainError):
            model.get_inverse_butler_volmer(None, None, "not a domain")


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

    def test_creation_lead_acid(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.LeadAcidReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n)
        j0_p = model.get_exchange_current_densities(self.c_e_p)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_creation_lithium_ion(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.LithiumIonReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n, self.c_s_n_surf)
        j0_p = model.get_exchange_current_densities(self.c_e_p, self.c_s_p_surf)
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters_lead_acid(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.LeadAcidReaction(param)
        j0_n = model.get_exchange_current_densities(self.c_e_n)
        j0_p = model.get_exchange_current_densities(self.c_e_p)
        # Process parameters
        parameter_values = model.default_parameter_values
        j0_n = parameter_values.process_symbol(j0_n)
        j0_p = parameter_values.process_symbol(j0_p)
        # Test
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_p.pre_order()]

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

    def test_discretisation_lead_acid(self):
        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.LeadAcidReaction(param)
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
        model = pybamm.interface.LeadAcidReaction(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_exchange_current_densities(None, "not a domain")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
