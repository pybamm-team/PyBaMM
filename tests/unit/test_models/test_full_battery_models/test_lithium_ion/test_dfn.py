#
# Tests for the lithium-ion DFN model
#
from tests import TestCase
import pybamm
import unittest
from tests import BaseUnitTestLithiumIon


class TestDFN(BaseUnitTestLithiumIon, TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
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

    def test_well_posed_constant_double_sei_layer(self):
        options = {"SEI": "constant", "double SEI layer": "true"}
        self.check_well_posedness(options)

    def test_well_posed_reaction_limited_growth_double_sei_layer(self):
        options = {"SEI": "reaction limited", "double SEI layer": "true"}
        self.check_well_posedness(options)

    def test_well_posed_electron_migration_limited_growth_double_sei_layer(self):
        options = {"SEI": "electron-migration limited", "double SEI layer": "true"}
        self.check_well_posedness(options)

    def test_well_posed_ec_reaction_limited_growth_double_sei_layer(self):
        options = {"SEI": "ec reaction limited", "double SEI layer": "true"}
        self.check_well_posedness(options)

    def test_well_posed_ec_reaction_limited_cracks_growth_double_sei_layer(self):
        options = {
            "SEI": "ec reaction limited",
            "double SEI layer": "true",
            "SEI on cracks": "true",
        }
        self.check_well_posedness(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
