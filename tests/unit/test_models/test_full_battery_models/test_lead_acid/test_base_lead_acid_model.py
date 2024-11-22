#
# Tests for the base lead acid model class
#
import pybamm
import pytest


class TestBaseLeadAcidModel:
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
