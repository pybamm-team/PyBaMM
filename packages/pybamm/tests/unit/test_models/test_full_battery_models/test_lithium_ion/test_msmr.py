#
# Tests for the lithium-ion MSMR model
#
import pybamm


class TestMSMR:
    def test_well_posed(self):
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        model.check_well_posedness()

    def test_well_posed_half_cell(self):
        model = pybamm.lithium_ion.MSMR(
            {"working electrode": "positive", "number of MSMR reactions": ("none", "6")}
        )
        model.check_well_posedness()

    def test_build_false(self):
        model = pybamm.lithium_ion.MSMR(
            {"number of MSMR reactions": ("6", "4")}, build=False
        )
        assert not model._built
