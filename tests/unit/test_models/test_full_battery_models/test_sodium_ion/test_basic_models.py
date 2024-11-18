#
# Tests for the basic sodium-ion models
#
import pybamm


class TestBasicModels:
    def test_dfn_well_posed(self):
        model = pybamm.sodium_ion.BasicDFN()
        model.check_well_posedness()

    def test_default_parameters(self):
        model = pybamm.sodium_ion.BasicDFN()
        assert "Chayambuka2022" in model.default_parameter_values["citations"]
