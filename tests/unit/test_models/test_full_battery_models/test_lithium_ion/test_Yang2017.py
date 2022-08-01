#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest
import numpy as np


class TestYang2017(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.Yang2017()
        model.check_well_posedness()

    def test_default_parameter_values(self):
        model = pybamm.lithium_ion.Yang2017()
        parameter_values = pybamm.ParameterValues({
            "chemistry": "lithium_ion",
            "cell": "LGM50_Chen2020",
            "negative electrode": "graphite_Chen2020",
            "separator": "separator_Chen2020",
            "positive electrode": "nmc_Chen2020",
            "electrolyte": "lipf6_Nyman2008",
            "experiment": "1C_discharge_from_full_Chen2020",
            "sei": "example",
            "lithium plating": "okane2020_Li_plating",
        })
        for key, value in parameter_values.items():
            if not isinstance(value, tuple):
                np.testing.assert_array_equal(
                    value, model.default_parameter_values[key]
                )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
