#
# Tests for the lithium-ion DFN model
#
import pybamm


class TestYang2017:
    def test_well_posed(self):
        model = pybamm.lithium_ion.Yang2017()
        model.check_well_posedness()
