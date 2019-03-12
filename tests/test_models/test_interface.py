#
# Tests for the electrode-electrolyte interface equations
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np
import os


class TestHomogeneousReaction(unittest.TestCase):
    def test_set_parameters(self):
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv",
            {
                "Typical current density": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            },
        )

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        rxn = pybamm.interface.homogeneous_reaction(whole_cell)
        processed_rxn = param.process_symbol(rxn)

        # rxn (a concatenation of functions of scalars and parameters) should get
        # discretised to a concantenation of functions of scalars
        self.assertIsInstance(processed_rxn, pybamm.Concatenation)
        self.assertFalse(
            any(
                [
                    isinstance(x, pybamm.Parameter)
                    for x in processed_rxn.children[0].pre_order()
                ]
            )
        )
        self.assertFalse(
            any(
                [
                    isinstance(x, pybamm.Parameter)
                    for x in processed_rxn.children[2].pre_order()
                ]
            )
        )
        self.assertEqual(processed_rxn.children[0].domain, ["negative electrode"])
        self.assertEqual(processed_rxn.children[1].domain, ["separator"])
        self.assertEqual(processed_rxn.children[2].domain, ["positive electrode"])

    def test_discretisation(self):
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv",
            {
                "Typical current density": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            },
        )
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        whole_cell = ["negative electrode", "separator", "positive electrode"]

        rxn = pybamm.interface.homogeneous_reaction(whole_cell)

        param_rxn = param.process_symbol(rxn)
        processed_rxn = disc.process_symbol(param_rxn)

        submesh = disc.mesh.combine_submeshes(*whole_cell)

        # processed_rxn should be a concatenation with the right shape
        self.assertIsInstance(processed_rxn, pybamm.Concatenation)
        self.assertEqual(processed_rxn.evaluate(0, None).shape, submesh.nodes.shape)

        # test values
        l_n = param.process_symbol(pybamm.standard_parameters.l_n)
        l_p = param.process_symbol(pybamm.standard_parameters.l_p)
        npts_n = mesh["negative electrode"].npts
        npts_s = mesh["separator"].npts
        np.testing.assert_array_equal(
            (l_n * processed_rxn).evaluate(0, None)[:npts_n], 1
        )
        np.testing.assert_array_equal(
            processed_rxn.evaluate(0, None)[npts_n : npts_n + npts_s], 0
        )
        np.testing.assert_array_equal(
            (l_p * processed_rxn).evaluate(0, None)[npts_n + npts_s :], -1
        )

    def test_disc_for_scalars(self):
        param = pybamm.ParameterValues(
            "input/parameters/lithium-ion/parameters/LCO.csv",
            {
                "Typical current density": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            },
        )
        disc = get_discretisation_for_testing()

        j_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        j_p = pybamm.interface.homogeneous_reaction(["positive electrode"])

        param_j_n = param.process_symbol(j_n)
        param_j_p = param.process_symbol(j_p)

        processed_j_n = disc.process_symbol(param_j_n)
        processed_j_p = disc.process_symbol(param_j_p)

        # test values
        l_n = param.process_symbol(pybamm.standard_parameters.l_n)
        l_p = param.process_symbol(pybamm.standard_parameters.l_p)

        np.testing.assert_array_equal((processed_j_n * l_n).evaluate(0, None), 1)
        np.testing.assert_array_equal((processed_j_p * l_p).evaluate(0, None), -1)

    def test_failure(self):
        with self.assertRaises(pybamm.DomainError):
            pybamm.interface.homogeneous_reaction(["not a domain"])


class TestButlerVolmer(unittest.TestCase):
    def setUp(self):
        self.cn = pybamm.Variable("concentration", domain=["negative electrode"])
        self.cs = pybamm.Variable("concentration", domain=["separator"])
        self.cp = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c = pybamm.Concatenation(self.cn, self.cs, self.cp)
        self.Delta_phin = pybamm.Variable(
            "potential difference", domain=["negative electrode"]
        )
        self.Delta_phis = pybamm.Variable("potential difference", domain=["separator"])
        self.Delta_phip = pybamm.Variable(
            "potential difference", domain=["positive electrode"]
        )
        self.Delta_phi = pybamm.Concatenation(
            self.Delta_phin, self.Delta_phis, self.Delta_phip
        )
        self.cn_surf = pybamm.Variable(
            "surface concentration", domain=["negative electrode"]
        )
        self.cp_surf = pybamm.Variable(
            "surface concentration", domain=["positive electrode"]
        )
        self.c_surf = pybamm.Concatenation(self.cn_surf, self.cp_surf)

        self.param = pybamm.standard_parameters
        self.param.__dict__.update(pybamm.standard_parameters_lead_acid.__dict__)

    def tearDown(self):
        del self.cn
        del self.cs
        del self.cp
        del self.c
        del self.Delta_phin
        del self.Delta_phis
        del self.Delta_phip
        del self.Delta_phi
        del self.cn_surf
        del self.cp_surf
        del self.c_surf
        del self.param

    def test_creation(self):
        # negative electrode passes, returns Multiplication
        bv = pybamm.interface.butler_volmer(self.param, self.cn, self.Delta_phin)
        self.assertIsInstance(bv, pybamm.Multiplication)
        self.assertEqual(bv.domain, ["negative electrode"])

        # positive electrode passes, returns Multiplication
        bv = pybamm.interface.butler_volmer(self.param, self.cp, self.Delta_phip)
        self.assertIsInstance(bv, pybamm.Multiplication)
        self.assertEqual(bv.domain, ["positive electrode"])

        # whole cell domain passes, returns concatenation
        bv = pybamm.interface.butler_volmer(self.param, self.c, self.Delta_phi)
        self.assertIsInstance(bv, pybamm.Concatenation)
        self.assertEqual(
            bv.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # c and Delta_phi without domain, domain gets input
        c = pybamm.Variable("concentration", domain=[])
        Delta_phi = pybamm.Variable("potential", domain=[])
        bv = pybamm.interface.butler_volmer(
            self.param, c, Delta_phi, domain=["negative electrode"]
        )
        self.assertIsInstance(bv, pybamm.Multiplication)
        self.assertEqual(bv.domain, [])

    def test_failures(self):
        # no way of determining domain
        c = pybamm.Variable("concentration", domain=[])
        Delta_phi = pybamm.Variable("potential", domain=[])
        with self.assertRaisesRegex(ValueError, "domain cannot be None"):
            pybamm.interface.butler_volmer(self.param, c, Delta_phi)

        # can't unpack concatenations
        with self.assertRaisesRegex(
            TypeError, "c_e and Delta_phi must both be Concatenations"
        ):
            pybamm.interface.butler_volmer(
                self.param,
                self.cn,
                self.Delta_phin,
                domain=["negative electrode", "separator", "positive electrode"],
            )
        with self.assertRaisesRegex(TypeError, "ck_surf must be a Concatenation"):
            pybamm.interface.butler_volmer(
                self.param,
                self.c,
                self.Delta_phi,
                self.cn_surf,
                domain=["negative electrode", "separator", "positive electrode"],
            )

        # bad domain
        with self.assertRaises(pybamm.DomainError):
            pybamm.interface.butler_volmer(
                self.param, c, Delta_phi, domain=["not a domain"]
            )

    def test_creation_with_particles(self):
        # negative electrode passes, returns Multiplication
        bv = pybamm.interface.butler_volmer(
            self.param, self.cn, self.Delta_phin, self.cn_surf
        )
        self.assertIsInstance(bv, pybamm.Multiplication)
        self.assertEqual(bv.domain, ["negative electrode"])

        # positive electrode passes, returns Multiplication
        bv = pybamm.interface.butler_volmer(
            self.param, self.cp, self.Delta_phip, self.cp_surf
        )
        self.assertIsInstance(bv, pybamm.Multiplication)
        self.assertEqual(bv.domain, ["positive electrode"])

        # whole cell domain passes, returns concatenation
        bv = pybamm.interface.butler_volmer(
            self.param, self.c, self.Delta_phi, self.c_surf
        )
        self.assertIsInstance(bv, pybamm.Concatenation)
        self.assertEqual(
            bv.domain, ["negative electrode", "separator", "positive electrode"]
        )

    def test_set_parameters(self):
        bv = pybamm.interface.butler_volmer(self.param, self.c, self.Delta_phi)
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        proc_bv = parameter_values.process_symbol(bv)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_bv.pre_order()]

        # with particles
        bv = pybamm.interface.butler_volmer(
            self.param, self.c, self.Delta_phi, self.c_surf
        )
        proc_bv = parameter_values.process_symbol(bv)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_bv.pre_order()]

    def test_discretisation(self):
        bv_n = pybamm.interface.butler_volmer(self.param, self.cn, self.Delta_phin)
        bv_p = pybamm.interface.butler_volmer(self.param, self.cp, self.Delta_phip)

        # process parameters
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        param_bv_n = parameter_values.process_symbol(bv_n)
        param_bv_p = parameter_values.process_symbol(bv_p)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        variables = [self.cn, self.cp, self.Delta_phin, self.Delta_phip]
        disc.set_variable_slices(variables)
        processed_bv_n = disc.process_symbol(param_bv_n)
        processed_bv_p = disc.process_symbol(param_bv_p)

        submesh = np.concatenate(
            [mesh["negative electrode"].nodes, mesh["positive electrode"].nodes]
        )
        y = np.concatenate([submesh ** 2, submesh ** 3])

        # should evaluate to vectors with the right shape
        self.assertEqual(
            processed_bv_n.evaluate(None, y).shape,
            mesh["negative electrode"].nodes.shape,
        )
        self.assertEqual(
            processed_bv_p.evaluate(None, y).shape,
            mesh["positive electrode"].nodes.shape,
        )

    def test_discretisation_with_particles(self):
        bv_n = pybamm.interface.butler_volmer(
            self.param, self.cn, self.Delta_phin, self.cn_surf
        )
        bv_p = pybamm.interface.butler_volmer(
            self.param, self.cp, self.Delta_phip, self.cp_surf
        )

        # process parameters
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        param_bv_n = parameter_values.process_symbol(bv_n)
        param_bv_p = parameter_values.process_symbol(bv_p)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        variables = [
            self.cn,
            self.cp,
            self.Delta_phin,
            self.Delta_phip,
            self.cn_surf,
            self.cp_surf,
        ]
        disc.set_variable_slices(variables)
        processed_bv_n = disc.process_symbol(param_bv_n)
        processed_bv_p = disc.process_symbol(param_bv_p)

        submesh = np.concatenate(
            [mesh["negative electrode"].nodes, mesh["positive electrode"].nodes]
        )
        y = np.concatenate([submesh ** 2, submesh ** 3, submesh])

        # should evaluate to vectors with the right shape
        self.assertEqual(
            processed_bv_n.evaluate(None, y).shape,
            mesh["negative electrode"].nodes.shape,
        )
        self.assertEqual(
            processed_bv_p.evaluate(None, y).shape,
            mesh["positive electrode"].nodes.shape,
        )

    def test_discretisation_whole(self):
        bv_whole = pybamm.interface.butler_volmer(self.param, self.c, self.Delta_phi)

        # process parameters
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        param_bv_whole = parameter_values.process_symbol(bv_whole)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        variables = [self.c, self.Delta_phi]
        disc.set_variable_slices(variables)
        processed_bv_whole = disc.process_symbol(param_bv_whole)

        # test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh.combine_submeshes(*whole_cell)
        y = np.concatenate([whole_cell_mesh.nodes ** 2, whole_cell_mesh.nodes ** 3])
        self.assertEqual(
            processed_bv_whole.evaluate(None, y).shape, whole_cell_mesh.nodes.shape
        )

    def test_discretisation_whole_with_particles(self):
        bv_whole = pybamm.interface.butler_volmer(
            self.param, self.c, self.Delta_phi, self.c_surf
        )

        # process parameters
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
        parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
        param_bv_whole = parameter_values.process_symbol(bv_whole)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        variables = [self.c, self.Delta_phi, self.c_surf]
        disc.set_variable_slices(variables)
        processed_bv_whole = disc.process_symbol(param_bv_whole)

        # test
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh.combine_submeshes(*whole_cell)
        submesh = np.concatenate(
            [mesh["negative electrode"].nodes, mesh["positive electrode"].nodes]
        )
        y = np.concatenate(
            [whole_cell_mesh.nodes ** 2, whole_cell_mesh.nodes ** 3, submesh]
        )
        self.assertEqual(
            processed_bv_whole.evaluate(None, y).shape, whole_cell_mesh.nodes.shape
        )


class TestExchangeCurrentDensity(unittest.TestCase):
    def setUp(self):
        self.cn = pybamm.Variable("concentration", domain=["negative electrode"])
        self.cs = pybamm.Variable("concentration", domain=["separator"])
        self.cp = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c = pybamm.Concatenation(self.cn, self.cs, self.cp)
        self.cn_surf = pybamm.Variable(
            "surface concentration", domain=["negative electrode"]
        )
        self.cp_surf = pybamm.Variable(
            "surface concentration", domain=["positive electrode"]
        )

    def tearDown(self):
        del self.cn
        del self.cs
        del self.cp
        del self.c
        del self.cn_surf
        del self.cp_surf

    def test_creation(self):
        # Concentration without domain passes
        c = pybamm.Variable("c")
        m = pybamm.standard_parameters.m_n
        pybamm.interface.exchange_current_density(c, domain=["negative electrode"])
        pybamm.interface.exchange_current_density(c, domain=["positive electrode"])

        # Concentration with correct domain passes
        j0n = pybamm.interface.exchange_current_density(self.cn)
        j0p = pybamm.interface.exchange_current_density(self.cp)
        self.assertEqual(j0n.domain, ["negative electrode"])
        self.assertEqual(j0p.domain, ["positive electrode"])

        # Surface concentration passes
        j0n = pybamm.interface.exchange_current_density(self.cn, self.cn_surf)
        j0p = pybamm.interface.exchange_current_density(self.cp, self.cp_surf)
        self.assertEqual(j0n.domain, ["negative electrode"])
        self.assertEqual(j0p.domain, ["positive electrode"])

        # Concentration without domain or with "not a domain" fails
        c = pybamm.Variable("concentration", domain=[])
        with self.assertRaises(ValueError):
            pybamm.interface.exchange_current_density(c)
        with self.assertRaises(KeyError):
            pybamm.interface.exchange_current_density(c, domain=["not a domain"])

    def test_set_parameters(self):
        param = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"Typical current density": 1}
        )
        # without surface concentration
        j0n = pybamm.interface.exchange_current_density(self.cn)
        j0p = pybamm.interface.exchange_current_density(self.cp)
        proc_j0n = param.process_symbol(j0n)
        proc_j0p = param.process_symbol(j0p)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_j0n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_j0p.pre_order()]
        # with surface concentration
        j0n = pybamm.interface.exchange_current_density(self.cn, self.cn_surf)
        j0p = pybamm.interface.exchange_current_density(self.cp, self.cp_surf)
        proc_j0n = param.process_symbol(j0n)
        proc_j0p = param.process_symbol(j0p)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_j0n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in proc_j0p.pre_order()]

    def test_discretisation(self):
        # create exchange-current densities
        j0n = pybamm.interface.exchange_current_density(self.cn)
        j0p = pybamm.interface.exchange_current_density(self.cp)

        # process parameters
        param = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"Typical current density": 1}
        )
        param_j0n = param.process_symbol(j0n)
        param_j0p = param.process_symbol(j0p)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        variables = [self.cn, self.cp]
        disc.set_variable_slices(variables)
        processed_j0n = disc.process_symbol(param_j0n)
        processed_j0p = disc.process_symbol(param_j0p)

        submesh = np.concatenate(
            [mesh["negative electrode"].nodes, mesh["positive electrode"].nodes]
        )
        y = submesh ** 2
        # should evaluate to vectors with the right shape
        self.assertEqual(
            processed_j0n.evaluate(y=y).shape, mesh["negative electrode"].nodes.shape
        )
        self.assertEqual(
            processed_j0p.evaluate(y=y).shape, mesh["positive electrode"].nodes.shape
        )

    def test_discretisation_surface_conc(self):
        # create exchange-current densities
        j0n = pybamm.interface.exchange_current_density(self.cn, self.cn_surf)
        j0p = pybamm.interface.exchange_current_density(self.cp, self.cp_surf)

        # process parameters
        param = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"Typical current density": 1}
        )
        param_j0n = param.process_symbol(j0n)
        param_j0p = param.process_symbol(j0p)

        # discretise
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        variables = [self.cn, self.cp, self.cn_surf, self.cp_surf]
        disc.set_variable_slices(variables)
        processed_j0n = disc.process_symbol(param_j0n)
        processed_j0p = disc.process_symbol(param_j0p)

        submesh = np.concatenate(
            [
                mesh["negative electrode"].nodes,
                mesh["positive electrode"].nodes,
                mesh["negative electrode"].nodes,
                mesh["positive electrode"].nodes,
            ]
        )
        y = submesh ** 2
        # should evaluate to vectors with the right shape
        self.assertEqual(
            processed_j0n.evaluate(y=y).shape, mesh["negative electrode"].nodes.shape
        )
        self.assertEqual(
            processed_j0p.evaluate(y=y).shape, mesh["positive electrode"].nodes.shape
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
