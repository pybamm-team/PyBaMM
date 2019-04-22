#
# Tests for the electrode-electrolyte interface equations
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestHomogeneousReaction(unittest.TestCase):
    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_vars = model.get_homogeneous_interfacial_current()
        parameter_values = model.default_parameter_values

        processed_j_vars = {
            name: parameter_values.process_symbol(var) for name, var in j_vars.items()
        }
        j = processed_j_vars["Interfacial current density"]

        self.assertIsInstance(j, pybamm.Concatenation)
        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j.children[0].pre_order()])
        )
        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j.children[2].pre_order()])
        )
        self.assertEqual(j.children[0].domain, ["negative electrode"])
        self.assertEqual(j.children[1].domain, ["separator"])
        self.assertEqual(j.children[2].domain, ["positive electrode"])

    def test_discretisation(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_vars = model.get_homogeneous_interfacial_current()
        parameter_values = model.default_parameter_values

        processed_vars = {
            name: disc.process_symbol(parameter_values.process_symbol(var))
            for name, var in j_vars.items()
        }
        j = processed_vars["Interfacial current density"]
        j_n = processed_vars["Negative electrode interfacial current density"]
        j_p = processed_vars["Positive electrode interfacial current density"]

        submesh = disc.mesh.combine_submeshes(*j.domain)

        self.assertIsInstance(j, pybamm.Concatenation)
        self.assertEqual(j.evaluate(0, None).shape, submesh[0].nodes.shape)

        # test values
        l_n = parameter_values.process_symbol(param.l_n)
        l_p = parameter_values.process_symbol(param.l_p)
        npts_n = mesh["negative electrode"][0].npts
        npts_s = mesh["separator"][0].npts
        np.testing.assert_array_equal((l_n * j).evaluate(0, None)[:npts_n], 1)
        np.testing.assert_array_equal(j.evaluate(0, None)[npts_n : npts_n + npts_s], 0)
        np.testing.assert_array_equal(
            (l_p * j).evaluate(0, None)[npts_n + npts_s :], -1
        )
        np.testing.assert_array_equal((l_n * j_n).evaluate(0, None), 1)
        np.testing.assert_array_equal((l_p * j_p).evaluate(0, None), -1)

    def test_simplify_constant_current(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j_vars = model.get_homogeneous_interfacial_current()
        parameter_values = model.default_parameter_values

        processed_vars = {
            name: disc.process_symbol(parameter_values.process_symbol(var))
            for name, var in j_vars.items()
        }
        j = processed_vars["Interfacial current density"]

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


class TestButlerVolmer(unittest.TestCase):
    def setUp(self):
        eta_r_n = pybamm.Variable("overpotential", domain=["negative electrode"])
        eta_r_p = pybamm.Variable("overpotential", domain=["positive electrode"])
        j0_n = pybamm.Variable("exchange-curr density", domain=["negative electrode"])
        j0_p = pybamm.Variable("exchange-curr density", domain=["positive electrode"])

        self.variables = {
            "Negative reaction overpotential": eta_r_n,
            "Positive reaction overpotential": eta_r_p,
            "Negative electrode exchange-current density": j0_n,
            "Positive electrode exchange-current density": j0_p,
        }

    def tearDown(self):
        del self.variables

    def test_creation(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        bv_vars = model.get_interfacial_current_butler_volmer(self.variables)
        bv_n = bv_vars["Negative electrode interfacial current density"]
        bv_p = bv_vars["Positive electrode interfacial current density"]
        bv = bv_vars["Interfacial current density"]

        # negative electrode Butler-Volmer is Multiplication
        self.assertIsInstance(bv_n, pybamm.Multiplication)
        self.assertEqual(bv_n.domain, ["negative electrode"])

        # positive electrode Butler-Volmer is Multiplication
        self.assertIsInstance(bv_p, pybamm.Multiplication)
        self.assertEqual(bv_p.domain, ["positive electrode"])

        # whole cell domain Butler-Volmer is concatenation
        self.assertIsInstance(bv, pybamm.Concatenation)
        self.assertEqual(
            bv.domain, ["negative electrode", "separator", "positive electrode"]
        )

    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        bv_vars = model.get_interfacial_current_butler_volmer(self.variables)

        # Process parameters
        parameter_values = model.default_parameter_values
        processed_vars = {
            name: parameter_values.process_symbol(var) for name, var in bv_vars.items()
        }
        bv_n = processed_vars["Negative electrode interfacial current density"]
        bv_p = processed_vars["Positive electrode interfacial current density"]
        bv = processed_vars["Interfacial current density"]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in bv.pre_order()]

    def test_discretisation(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        bv_vars = model.get_interfacial_current_butler_volmer(self.variables)

        # Process parameters and discretise
        parameter_values = model.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(self.variables.values())

        processed_vars = {
            name: disc.process_symbol(parameter_values.process_symbol(var))
            for name, var in bv_vars.items()
        }
        bv_n = processed_vars["Negative electrode interfacial current density"]
        bv_p = processed_vars["Positive electrode interfacial current density"]
        bv = processed_vars["Interfacial current density"]

        # test butler-volmer in each electrode
        submesh = np.concatenate(
            [mesh["negative electrode"][0].nodes, mesh["positive electrode"][0].nodes]
        )
        y = np.concatenate([submesh ** 2, submesh ** 3])
        self.assertEqual(
            bv_n.evaluate(None, y).shape, mesh["negative electrode"][0].nodes.shape
        )
        self.assertEqual(
            bv_p.evaluate(None, y).shape, mesh["positive electrode"][0].nodes.shape
        )

        # test concatenated butlver-volmer
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh.combine_submeshes(*whole_cell)
        self.assertEqual(bv.evaluate(None, y).shape, whole_cell_mesh[0].nodes.shape)


class TestExchangeCurrentDensity(unittest.TestCase):
    def setUp(self):
        c_e_n = pybamm.Variable("concentration", domain=["negative electrode"])
        c_e_s = pybamm.Variable("concentration", domain=["separator"])
        c_e_p = pybamm.Variable("concentration", domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        c_s_n = pybamm.Variable("surface conc", domain=["negative electrode"])
        c_s_p = pybamm.Variable("surface conc", domain=["positive electrode"])
        self.variables = {
            "Electrolyte concentration": c_e,
            "Negative particle concentration": c_s_n,
            "Positive particle concentration": c_s_p,
        }

    def tearDown(self):
        del self.variables

    def test_creation(self):
        # With intercalation
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j0_vars = model.get_exchange_current_densities(self.variables)
        j0_n = j0_vars["Negative electrode exchange-current density"]
        j0_p = j0_vars["Positive electrode exchange-current density"]
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

        # Without intercalation
        param_la = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.InterfacialCurrent(param_la)
        j0_vars = model.get_exchange_current_densities(self.variables, False)
        j0_n = j0_vars["Negative electrode exchange-current density"]
        j0_p = j0_vars["Positive electrode exchange-current density"]
        self.assertEqual(j0_n.domain, ["negative electrode"])
        self.assertEqual(j0_p.domain, ["positive electrode"])

    def test_set_parameters(self):
        # With intercalation
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j0_vars = model.get_exchange_current_densities(self.variables)
        # Process parameters
        parameter_values = model.default_parameter_values
        processed_vars = {
            name: parameter_values.process_symbol(var) for name, var in j0_vars.items()
        }
        j0_n = processed_vars["Negative electrode exchange-current density"]
        j0_p = processed_vars["Positive electrode exchange-current density"]
        # Test
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_p.pre_order()]

        # Without intercalation
        param_la = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.InterfacialCurrent(param_la)
        # Process parameters
        processed_vars = {
            name: parameter_values.process_symbol(var) for name, var in j0_vars.items()
        }
        j0_vars = model.get_exchange_current_densities(self.variables, False)
        j0_n = processed_vars["Negative electrode exchange-current density"]
        j0_p = processed_vars["Positive electrode exchange-current density"]
        # Test
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j0_p.pre_order()]

    def test_discretisation(self):
        # With intercalation
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.interface.InterfacialCurrent(param)
        j0_vars = model.get_exchange_current_densities(self.variables)
        # Process parameters and discretise
        parameter_values = model.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(self.variables.values())
        processed_vars = {
            name: disc.process_symbol(parameter_values.process_symbol(var))
            for name, var in j0_vars.items()
        }
        j0_n = processed_vars["Negative electrode exchange-current density"]
        j0_p = processed_vars["Positive electrode exchange-current density"]
        # Test
        submesh = np.concatenate(
            [
                mesh["negative electrode"][0].nodes,
                mesh["positive electrode"][0].nodes,
                mesh["negative electrode"][0].nodes,
                mesh["positive electrode"][0].nodes,
            ]
        )
        y = submesh ** 2
        # should evaluate to vectors with the right shape
        import ipdb

        ipdb.set_trace()
        self.assertEqual(
            j0_n.evaluate(y=y).shape, mesh["negative electrode"][0].nodes.shape
        )
        self.assertEqual(
            j0_p.evaluate(y=y).shape, mesh["positive electrode"][0].nodes.shape
        )

        # Without intercalation
        param_la = pybamm.standard_parameters_lead_acid
        model = pybamm.interface.InterfacialCurrent(param_la)
        # Process parameters
        processed_vars = {
            name: disc.process_symbol(parameter_values.process_symbol(var))
            for name, var in j0_vars.items()
        }
        j0_vars = model.get_exchange_current_densities(self.variables, False)
        j0_n = processed_vars["Negative electrode exchange-current density"]
        j0_p = processed_vars["Positive electrode exchange-current density"]
        # Test
        submesh = np.concatenate(
            [mesh["negative electrode"][0].nodes, mesh["positive electrode"][0].nodes]
        )
        y = submesh ** 2
        # should evaluate to vectors with the right shape
        self.assertEqual(
            j0_n.evaluate(y=y).shape, mesh["negative electrode"][0].nodes.shape
        )
        self.assertEqual(
            j0_p.evaluate(y=y).shape, mesh["positive electrode"][0].nodes.shape
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
