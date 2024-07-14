#
# Tests for the lithium-ion DFN model
#
import pybamm
import pytest
from tests import BaseUnitTestLithiumIon


class TestDFN(BaseUnitTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with pytest.raises(pybamm.OptionError, match="electrolyte conductivity"):
            pybamm.lithium_ion.DFN(options)

    def test_well_posed_size_distribution(self):
        options = {"particle size": "distribution"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_uniform_profile(self):
        options = {"particle size": "distribution", "particle": "uniform profile"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_tuple(self):
        options = {"particle size": ("single", "distribution")}
        self.check_well_posedness(options)

    def test_well_posed_current_sigmoid_ocp_with_psd(self):
        options = {
            "open-circuit potential": "current sigmoid",
            "particle size": "distribution",
        }
        self.check_well_posedness(options)

    def test_well_posed_wycisk_ocp_with_psd(self):
        options = {
            "open-circuit potential": "Wycisk",
            "particle size": "distribution",
        }
        self.check_well_posedness(options)

    def test_well_posed_wycisk_ocp_with_composite(self):
        options = {
            "open-circuit potential": (("Wycisk", "single"), "single"),
            "particle phases": ("2", "1"),
        }
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_explicit_power(self):
        options = {"operating mode": "explicit power"}
        self.check_well_posedness(options)

    def test_well_posed_external_circuit_explicit_resistance(self):
        options = {"operating mode": "explicit resistance"}
        self.check_well_posedness(options)

    def test_well_posed_msmr_with_psd(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "particle size": "distribution",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        self.check_well_posedness(options)
