#
# Tests for the electrode-electrolyte interface equations
#
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestButlerVolmer(unittest.TestCase):
    def setUp(self):
        self.delta_phi_s_n = pybamm.Variable(
            "surface potential difference", ["negative electrode"]
        )
        self.delta_phi_s_p = pybamm.Variable(
            "surface potential difference", ["positive electrode"]
        )
        c_e_n = pybamm.Variable("concentration", domain=["negative electrode"])
        c_e_s = pybamm.Variable("concentration", domain=["separator"])
        c_e_p = pybamm.Variable("concentration", domain=["positive electrode"])
        self.c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)
        self.c_s_n_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["negative particle"])
        )
        self.c_s_p_surf = pybamm.surf(
            pybamm.Variable("particle conc", domain=["positive particle"])
        )
        self.variables = {
            "Negative electrode surface potential difference": self.delta_phi_s_n,
            "Positive electrode surface potential difference": self.delta_phi_s_p,
            "Negative electrolyte concentration": c_e_n,
            "Positive electrolyte concentration": c_e_p,
            "Negative particle surface concentration": self.c_s_n_surf,
            "Positive particle surface concentration": self.c_s_p_surf,
            "Current collector current density": pybamm.Scalar(1),
            "Negative electrode open circuit potential": pybamm.Scalar(0),
            "Positive electrode open circuit potential": pybamm.Scalar(0),
        }

    def tearDown(self):
        del self.variables
        del self.c_e
        del self.c_s_n_surf
        del self.c_s_p_surf
        del self.delta_phi_s_n
        del self.delta_phi_s_p

    def test_creation(self):
        param = pybamm.standard_parameters_lithium_ion
        model_n = pybamm.interface.lithium_ion.ButlerVolmer(param, "Negative")
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density"
        ]
        model_p = pybamm.interface.lithium_ion.ButlerVolmer(param, "Positive")
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density"
        ]

        # negative electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_n, pybamm.Multiplication)
        self.assertEqual(j_n.domain, ["negative electrode"])

        # positive electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_p, pybamm.Multiplication)
        self.assertEqual(j_p.domain, ["positive electrode"])

    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        model_n = pybamm.interface.lithium_ion.ButlerVolmer(param, "Negative")
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density"
        ]
        model_p = pybamm.interface.lithium_ion.ButlerVolmer(param, "Positive")
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density"
        ]

        # Process parameters
        parameter_values = model_n.default_parameter_values
        j_n = parameter_values.process_symbol(j_n)
        j_p = parameter_values.process_symbol(j_p)
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j_n.pre_order()]
        [self.assertNotIsInstance(x, pybamm.Parameter) for x in j_p.pre_order()]

    def test_discretisation(self):
        param = pybamm.standard_parameters_lithium_ion
        model_n = pybamm.interface.lithium_ion.ButlerVolmer(param, "Negative")
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density"
        ]
        model_p = pybamm.interface.lithium_ion.ButlerVolmer(param, "Positive")
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density"
        ]
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)

        # Process parameters and discretise
        parameter_values = model_n.default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(
            [
                self.c_e,
                self.delta_phi_s_n,
                self.delta_phi_s_p,
                self.c_s_n_surf,
                self.c_s_p_surf,
            ]
        )

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

        # test concatenated butler-volmer
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh.combine_submeshes(*whole_cell)
        self.assertEqual(j.evaluate(None, y).shape, (whole_cell_mesh[0].npts, 1))

    def test_diff_c_e_lead_acid(self):

        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.lead_acid.ButlerVolmer(param, "Negative")
        model_p = pybamm.interface.lead_acid.ButlerVolmer(param, "Positive")
        parameter_values = model_n.default_parameter_values

        def j_n(c_e):
            eta_r_n = 1 - param.U_n(c_e)
            return model.get_butler_volmer(c_e, eta_r_n, ["negative electrode"])

        def j_p(c_e):
            eta_r_p = 1 - param.U_p(c_e)
            return model.get_butler_volmer(c_e, eta_r_p, ["positive electrode"])

        c_e = pybamm.Scalar(0.5)
        h = pybamm.Scalar(0.00001)

        # Analytical
        j_n_diff = parameter_values.process_symbol(j_n(c_e).diff(c_e))
        j_p_diff = parameter_values.process_symbol(j_p(c_e).diff(c_e))

        # Numerical
        j_n_FD = parameter_values.process_symbol(
            (j_n(c_e + h) - j_n(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(j_n_diff.evaluate(), j_n_FD.evaluate())
        j_p_FD = parameter_values.process_symbol(
            (j_p(c_e + h) - j_p(c_e - h)) / (2 * h)
        )
        self.assertAlmostEqual(j_p_diff.evaluate(), j_p_FD.evaluate())

    def test_diff_delta_phi_e_lead_acid(self):

        # With intercalation
        param = pybamm.standard_parameters_lead_acid
        model_n = pybamm.interface.lead_acid.ButlerVolmer(param, "Negative")
        model_p = pybamm.interface.lead_acid.ButlerVolmer(param, "Positive")
        parameter_values = model_n.default_parameter_values

        def j_n(delta_phi):
            return model.get_butler_volmer(1, delta_phi - 1, ["negative electrode"])

        def j_p(delta_phi):
            return model.get_butler_volmer(1, delta_phi - 1, ["positive electrode"])

        delta_phi = pybamm.Scalar(0.5)
        h = pybamm.Scalar(0.00001)

        # Analytical
        j_n_diff = parameter_values.process_symbol(j_n(delta_phi).diff(delta_phi))
        j_p_diff = parameter_values.process_symbol(j_p(delta_phi).diff(delta_phi))

        # Numerical
        j_n_FD = parameter_values.process_symbol(
            (j_n(delta_phi + h) - j_n(delta_phi - h)) / (2 * h)
        )
        self.assertAlmostEqual(j_n_diff.evaluate(), j_n_FD.evaluate())
        j_p_FD = parameter_values.process_symbol(
            (j_p(delta_phi + h) - j_p(delta_phi - h)) / (2 * h)
        )
        self.assertAlmostEqual(j_p_diff.evaluate(), j_p_FD.evaluate())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
