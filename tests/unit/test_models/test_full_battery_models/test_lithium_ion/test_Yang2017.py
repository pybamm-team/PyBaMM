#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest


class TestYang2017(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.Yang2017()
        model.check_well_posedness()

    def test_default_parameter_values(self):
        model = pybamm.lithium_ion.Yang2017()
        chemistry = pybamm.parameter_sets.Yang2017
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        self.assertDictEqual(
            parameter_values._dict_items, model.default_parameter_values._dict_items
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
