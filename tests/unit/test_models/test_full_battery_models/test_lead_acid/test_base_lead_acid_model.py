#
# Tests for the base lead acid model class
#
import pybamm
import pytest


class TestBaseLeadAcidModel:
    def test_default_geometry(self):
        model = pybamm.lead_acid.BaseModel({"dimensionality": 0})
        assert model.default_geometry["current collector"]["z"]["position"] == 1
        model = pybamm.lead_acid.BaseModel({"dimensionality": 1})
        assert model.default_geometry["current collector"]["z"]["min"] == 0
        model = pybamm.lead_acid.BaseModel({"dimensionality": 2})
        assert model.default_geometry["current collector"]["y"]["min"] == 0

    def test_incompatible_options(self):
        with pytest.raises(
            pybamm.OptionError,
            match="Lead-acid models can only have thermal effects if dimensionality is 0.",
        ):
            pybamm.lead_acid.BaseModel({"dimensionality": 1, "thermal": "lumped"})
        with pytest.raises(pybamm.OptionError, match="SEI"):
            pybamm.lead_acid.BaseModel({"SEI": "constant"})
        with pytest.raises(pybamm.OptionError, match="lithium plating"):
            pybamm.lead_acid.BaseModel({"lithium plating": "reversible"})
        with pytest.raises(pybamm.OptionError, match="MSMR"):
            pybamm.lead_acid.BaseModel(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                }
            )
