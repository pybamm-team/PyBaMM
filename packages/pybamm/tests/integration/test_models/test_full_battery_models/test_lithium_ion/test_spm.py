#
# Tests for the lithium-ion SPM model
#
import pytest

import pybamm
import tests
from tests import BaseIntegrationTestLithiumIon


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pybamm.lithium_ion.SPM

    def test_positive_electrode_degradation(self):
        options = {"positive electrode degradation": "true"}
        parameter_values = pybamm.ParameterValues("Zhuo2023")
        # Charging the cell from a full discharged state till it reaches 4.2 V
        parameter_values["Current function [A]"] = -3.35
        parameter_values["Upper voltage cut-off [V]"] = 4.2
        modeltest = tests.StandardModelTest(
            self.model(options), parameter_values=parameter_values
        )
        # The standard output tests assume a Fickian positive particle
        modeltest.test_all(skip_output_tests=True)
