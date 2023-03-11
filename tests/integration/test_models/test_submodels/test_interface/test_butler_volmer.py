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
            "surface potential difference [V]",
            ["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.delta_phi_s_p = pybamm.Variable(
            "surface potential difference [V]",
            ["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.c_e_n = pybamm.Variable(
            "concentration [mol.m-3]",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.c_e_p = pybamm.Variable(
            "concentration [mol.m-3]",
            domain=["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.c_s_n_surf = pybamm.Variable(
            "particle surface conc",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.c_s_p_surf = pybamm.Variable(
            "particle surface conc",
            domain=["positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        self.variables = {
            "Negative electrode surface potential difference [V]": self.delta_phi_s_n,
            "Positive electrode surface potential difference [V]": self.delta_phi_s_p,
            "Negative electrolyte concentration [mol.m-3]": self.c_e_n,
            "Positive electrolyte concentration [mol.m-3]": self.c_e_p,
            "Negative particle surface concentration [mol.m-3]": self.c_s_n_surf,
            "Positive particle surface concentration [mol.m-3]": self.c_s_p_surf,
            "Current collector current density [A.m-2]": pybamm.Scalar(1),
            "Negative electrode temperature [K]": 300,
            "Positive electrode temperature [K]": 300,
            "Negative electrode surface area to volume ratio [m-1]": 1 + 0 * self.c_e_n,
            "Positive electrode surface area to volume ratio [m-1]": 1 + 0 * self.c_e_p,
            "X-averaged negative electrode surface area to volume ratio [m-1]": 1,
            "X-averaged positive electrode surface area to volume ratio [m-1]": 1,
            "Negative electrode interface utilisation": 1,
            "Positive electrode interface utilisation": 1,
            "Negative electrode open-circuit potential [V]": pybamm.Scalar(0),
            "Positive electrode open-circuit potential [V]": pybamm.Scalar(0),
            "Sum of electrolyte reaction source terms [A.m-3]": pybamm.Scalar(1),
            "Sum of interfacial current densities [A.m-3]": pybamm.Scalar(1),
            "Sum of negative electrode volumetric"
            "interfacial current densities [A.m-3]": pybamm.Scalar(1),
            "Sum of positive electrode volumetric"
            "interfacial current densities [A.m-3]": pybamm.Scalar(1),
            "Sum of negative electrode electrolyte reaction source terms [A.m-3]": 1,
            "Sum of positive electrode electrolyte reaction source terms [A.m-3]": 1,
            "Sum of x-averaged negative electrode electrolyte "
            "reaction source terms [A.m-3]": 1,
            "Sum of x-averaged positive electrode electrolyte "
            "reaction source terms [A.m-3]": 1,
        }

    def tearDown(self):
        del self.variables
        del self.c_e_n
        del self.c_e_p
        del self.c_s_n_surf
        del self.c_s_p_surf
        del self.delta_phi_s_n
        del self.delta_phi_s_p

    def test_creation(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "negative",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density [A.m-2]"
        ]
        model_p = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "positive",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density [A.m-2]"
        ]

        # negative electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_n, pybamm.Multiplication)
        self.assertEqual(j_n.domain, ["negative electrode"])

        # positive electrode Butler-Volmer is Multiplication
        self.assertIsInstance(j_p, pybamm.Multiplication)
        self.assertEqual(j_p.domain, ["positive electrode"])

    def test_set_parameters(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "negative",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density [A.m-2]"
        ]
        model_p = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "positive",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density [A.m-2]"
        ]

        # Process parameters
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values

        j_n = parameter_values.process_symbol(j_n)
        j_p = parameter_values.process_symbol(j_p)
        # Test
        for x in j_n.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)
        for x in j_p.pre_order():
            self.assertNotIsInstance(x, pybamm.Parameter)

    def test_discretisation(self):
        param = pybamm.LithiumIonParameters()
        model_n = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "negative",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_n = model_n.get_coupled_variables(self.variables)[
            "Negative electrode interfacial current density [A.m-2]"
        ]
        model_p = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "positive",
            "lithium-ion main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
                "particle size": "single",
            },
            "primary",
        )
        j_p = model_p.get_coupled_variables(self.variables)[
            "Positive electrode interfacial current density [A.m-2]"
        ]
        j = pybamm.concatenation(j_n, pybamm.PrimaryBroadcast(0, ["separator"]), j_p)

        # Process parameters and discretise
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        disc.set_variable_slices(
            [
                self.c_e_n,
                self.c_e_p,
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
            [mesh["negative electrode"].nodes, mesh["positive electrode"].nodes]
        )
        y = np.concatenate([submesh**2, submesh**3, submesh**4])
        self.assertEqual(
            j_n.evaluate(None, y).shape, (mesh["negative electrode"].npts, 1)
        )
        self.assertEqual(
            j_p.evaluate(None, y).shape, (mesh["positive electrode"].npts, 1)
        )

        # test concatenated butler-volmer
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        whole_cell_mesh = disc.mesh[whole_cell]
        self.assertEqual(j.evaluate(None, y).shape, (whole_cell_mesh.npts, 1))

    def test_diff_c_e_lead_acid(self):
        # With intercalation
        param = pybamm.LeadAcidParameters()
        model_n = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "negative",
            "lead-acid main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
            },
            "primary",
        )
        model_p = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "positive",
            "lead-acid main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
            },
            "primary",
        )
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values

        def j_n(c_e):
            variables = {
                **self.variables,
                "Negative electrode surface potential difference [V]": 1,
                "Negative electrolyte concentration [mol.m-3]": c_e,
            }
            return model_n.get_coupled_variables(variables)[
                "Negative electrode interfacial current density [A.m-2]"
            ].orphans[0]

        def j_p(c_e):
            variables = {
                **self.variables,
                "Positive electrode surface potential difference [V]": 1,
                "Positive electrolyte concentration [mol.m-3]": c_e,
            }
            return model_p.get_coupled_variables(variables)[
                "Positive electrode interfacial current density [A.m-2]"
            ].orphans[0]

        c_e = pybamm.InputParameter("c_e")
        h = pybamm.Scalar(0.00001)

        # Analytical
        j_n_diff = parameter_values.process_symbol(j_n(c_e).diff(c_e))
        j_p_diff = parameter_values.process_symbol(j_p(c_e).diff(c_e))

        # Numerical
        j_n_FD = parameter_values.process_symbol(
            (j_n(c_e + h) - j_n(c_e - h)) / (2 * h)
        )
        np.testing.assert_almost_equal(
            j_n_diff.evaluate(inputs={"c_e": 0.5})
            / j_n_FD.evaluate(inputs={"c_e": 0.5}),
            1,
            decimal=5,
        )
        j_p_FD = parameter_values.process_symbol(
            (j_p(c_e + h) - j_p(c_e - h)) / (2 * h)
        )
        np.testing.assert_almost_equal(
            j_p_diff.evaluate(inputs={"c_e": 0.5})
            / j_p_FD.evaluate(inputs={"c_e": 0.5}),
            1,
            decimal=5,
        )

    def test_diff_delta_phi_e_lead_acid(self):
        # With intercalation
        param = pybamm.LeadAcidParameters()
        model_n = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "negative",
            "lead-acid main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
            },
            "primary",
        )
        model_p = pybamm.kinetics.SymmetricButlerVolmer(
            param,
            "positive",
            "lead-acid main",
            {
                "SEI film resistance": "none",
                "total interfacial current density as a state": "false",
            },
            "primary",
        )
        parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values

        def j_n(delta_phi):
            variables = {
                **self.variables,
                "Negative electrode surface potential difference [V]": delta_phi,
                "Negative electrolyte concentration [mol.m-3]": 1,
            }
            return model_n.get_coupled_variables(variables)[
                "Negative electrode interfacial current density [A.m-2]"
            ].orphans[0]

        def j_p(delta_phi):
            variables = {
                **self.variables,
                "Positive electrode surface potential difference [V]": delta_phi,
                "Positive electrolyte concentration [mol.m-3]": 1,
            }
            return model_p.get_coupled_variables(variables)[
                "Positive electrode interfacial current density [A.m-2]"
            ].orphans[0]

        delta_phi = pybamm.InputParameter("delta_phi")
        h = pybamm.Scalar(0.00001)

        # Analytical
        x = j_n(delta_phi)
        x.diff(delta_phi)
        j_n_diff = parameter_values.process_symbol(j_n(delta_phi).diff(delta_phi))
        j_p_diff = parameter_values.process_symbol(j_p(delta_phi).diff(delta_phi))

        # Numerical
        j_n_FD = parameter_values.process_symbol(
            (j_n(delta_phi + h) - j_n(delta_phi - h)) / (2 * h)
        )
        self.assertAlmostEqual(
            j_n_diff.evaluate(inputs={"delta_phi": 0.5})
            / j_n_FD.evaluate(inputs={"delta_phi": 0.5}),
            1,
            places=5,
        )
        j_p_FD = parameter_values.process_symbol(
            (j_p(delta_phi + h) - j_p(delta_phi - h)) / (2 * h)
        )
        self.assertAlmostEqual(
            j_p_diff.evaluate(inputs={"delta_phi": 0.5})
            / j_p_FD.evaluate(inputs={"delta_phi": 0.5}),
            1,
            places=5,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
