#
# Tests for the half-cell lithium-ion SPM model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIonHalfCell


class TestSPMHalfCell(BaseIntegrationTestLithiumIonHalfCell):
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
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Negative electrode Butler-Volmer transfer coefficient": 0.6,
                "Positive electrode Butler-Volmer transfer coefficient": 0.6,
            }
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_kinetics_linear(self):
        options = {"intercalation kinetics": "linear", "surface form": "algebraic"}
        self.run_basic_processing_test(options)

    def test_kinetics_mhc(self):
        options = {
            "intercalation kinetics": "Marcus-Hush-Chidsey",
            "surface form": "algebraic",
        }
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Negative electrode reorganization energy [eV]": 0.35,
                "Positive electrode reorganization energy [eV]": 0.35,
                "Positive electrode exchange-current density [A.m-2]": 5,
            }
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)
