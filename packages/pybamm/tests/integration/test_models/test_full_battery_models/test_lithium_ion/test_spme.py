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
