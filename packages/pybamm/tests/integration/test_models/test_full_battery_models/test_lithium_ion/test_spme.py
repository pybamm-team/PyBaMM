#
# Tests for the lithium-ion SPMe model
#
import pytest

import pybamm
import tests
from tests import BaseIntegrationTestLithiumIon


class TestSPMe(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.run_basic_processing_test(options)

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
