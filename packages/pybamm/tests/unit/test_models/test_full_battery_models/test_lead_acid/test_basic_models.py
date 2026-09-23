#
# Tests for the basic lead acid models
#
import pybamm


class TestBasicModels:
    def test_basic_full_lead_acid_well_posed(self):
        model = pybamm.lead_acid.BasicFull()
        model.check_well_posedness()
