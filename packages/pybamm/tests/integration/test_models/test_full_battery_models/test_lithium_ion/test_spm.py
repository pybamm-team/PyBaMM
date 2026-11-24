#
# Tests for the lithium-ion SPM model
#
import numpy as np
import pytest

import pybamm
import tests
from tests import BaseIntegrationTestLithiumIon


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPM

    def test_kinetics_asymmetric_butler_volmer(self):
        # SPM's own model-specific default sets "surface form" to "algebraic"
        # whenever "intercalation kinetics" is supplied.
        options = {
            "intercalation kinetics": "asymmetric Butler-Volmer",
            "surface form": "algebraic",
        }
        solver = pybamm.IDAKLUSolver(atol=1e-14, rtol=1e-14)

        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Negative electrode Butler-Volmer transfer coefficient": 0.6,
                "Positive electrode Butler-Volmer transfer coefficient": 0.6,
            }
        )
        self.run_basic_processing_test(
            options, parameter_values=parameter_values, solver=solver
        )

    def test_kinetics_linear(self):
        options = {"intercalation kinetics": "linear", "surface form": "algebraic"}
        self.run_basic_processing_test(options)

    def test_kinetics_mhc(self):
        options = {
            "intercalation kinetics": "Marcus-Hush-Chidsey",
            "surface form": "algebraic",
        }
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Negative electrode reorganization energy [eV]": 0.35,
                "Positive electrode reorganization energy [eV]": 0.35,
                "Positive electrode exchange-current density [A.m-2]": 5,
            }
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_kinetics_marcus(self):
        options = {
            "intercalation kinetics": "Marcus",
            "surface form": "algebraic",
        }
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Negative electrode reorganization energy [eV]": 0.35,
                "Positive electrode reorganization energy [eV]": 0.35,
                "Positive electrode exchange-current density [A.m-2]": 5,
            }
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_basic_processing_msmr(self):
        # SPM's own model-specific default sets "surface form" to "algebraic"
        # whenever "intercalation kinetics" is supplied.
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "intercalation kinetics": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "surface form": "algebraic",
        }
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    @pytest.mark.parametrize("dimensionality", [1, 2])
    def test_potential_pair_parallel_electrodes(self, dimensionality):
        # N identical electrodes in parallel at current I must behave like a single
        # electrode at I / N
        model = self.model(
            {
                "current collector": "potential pair",
                "dimensionality": dimensionality,
                "cell geometry": "pouch",
            }
        )
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        n_parallel, current = 10, 1

        solutions = []
        for n, I in [(1, current / n_parallel), (n_parallel, current)]:
            param = model.default_parameter_values
            param.update(
                {
                    "Number of electrodes connected in parallel to make a cell": n,
                    "Current function [A]": I,
                }
            )
            sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts)
            solutions.append(sim.solve([0, 600]))

        t = np.linspace(0, 600, 11)
        for name in ["Current collector current density [A.m-2]", "Voltage [V]"]:
            np.testing.assert_allclose(
                solutions[0][name](t), solutions[1][name](t), rtol=1e-3
            )
