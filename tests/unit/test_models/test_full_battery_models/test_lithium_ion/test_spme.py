#
# Tests for the lithium-ion SPMe model
#
import pybamm
from tests import BaseUnitTestLithiumIon
import pytest


class TestSPMe(BaseUnitTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPMe

    # def test_external_variables(self):
    #     # To Do: replace external variable with input
    #     # a concatenation
    #     model_options = {"external submodels": ["electrolyte diffusion"]}
    #     model = pybamm.lithium_ion.SPMe(model_options)
    #     self.assertEqual(
    #         model.external_variables[0],
    #         model.variables["Porosity times concentration"],
    #     )

    #     # a variable
    #     model_options = {"thermal": "lumped", "external submodels": ["thermal"]}
    #     model = pybamm.lithium_ion.SPMe(model_options)
    #     self.assertEqual(
    #         model.external_variables[0],
    #         model.variables["Volume-averaged cell temperature"],
    #     )

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with pytest.raises(pybamm.OptionError, match="electrolyte conductivity"):
            pybamm.lithium_ion.SPMe(options)

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.check_well_posedness(options)
