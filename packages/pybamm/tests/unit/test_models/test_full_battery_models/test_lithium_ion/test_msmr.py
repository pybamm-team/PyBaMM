#
# Tests for the lithium-ion MSMR model
#
import pybamm


class TestMSMR:
    def test_well_posed(self):
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        model.check_well_posedness()
