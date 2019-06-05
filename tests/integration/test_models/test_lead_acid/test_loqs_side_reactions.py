#
# Tests for the lead-acid LOQS model
#
import pybamm
import tests
import unittest


class TestLeadAcidLOQSWithSideReactions(unittest.TestCase):
    def test_basic_processing(self):
        options = {"capacitance": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_charge(self):
        options = {"capacitance": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Typical current [A]": -1, "Initial State of Charge": 0.5}
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_zero_current(self):
        options = {"capacitance": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
