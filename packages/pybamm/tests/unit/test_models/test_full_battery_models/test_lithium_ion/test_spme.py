"""Tests for the lithium-ion SPMe model."""

import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestSPMe(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.lithium_ion.SPMe(options)

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.check_well_posedness(options)
